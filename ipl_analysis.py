# ═══════════════════════════════════════════════════════════════
# IPL EDA Analysis — 5 Seasons (2019 to 2023)
# Questions: Toss impact | Phase analysis | Top batters & bowlers
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────
# STEP 1 — LOAD
# Change the path below to wherever your CSV file is
# ───────────────────────────────────────────────────────
FILE_PATH = 'ipl_data.csv'   # ← update this path

df = pd.read_csv(FILE_PATH, low_memory=False)

# Fix mixed-type season column (some seasons are "2020/21" strings)
df['season'] = df['season'].astype(str).str.strip()

# ───────────────────────────────────────────────────────
# STEP 2 — FILTER TO 5 SEASONS
# ───────────────────────────────────────────────────────
SEASONS = ['2019', '2020/21', '2021', '2022', '2023']
df = df[df['season'].isin(SEASONS)].copy()

# Drop no-result and tied matches (no clear winner to analyse)
df = df[~df['winner'].isin(['no result', 'tie'])].copy()

print(f"Loaded: {len(df):,} balls | {df['match_id'].nunique()} matches | {df['season'].nunique()} seasons")

# ───────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ───────────────────────────────────────────────────────

# Over numbers in Cricsheet are 0-indexed; convert to 1–20
df['over_num'] = df['over'] + 1

# Assign phase based on over number
def get_phase(o):
    if o <= 6:    return 'Powerplay\n(Overs 1–6)'
    elif o <= 15: return 'Middle Overs\n(Overs 7–15)'
    else:         return 'Death Overs\n(Overs 16–20)'

df['phase'] = df['over_num'].apply(get_phase)

# Wicket flag — ONLY credit the bowler for their own wickets
# Run-outs, retired hurt etc. do NOT count against the bowler
NON_BOWLER_WICKETS = ['run out', 'obstructing the field', 'retired hurt', 'retired out']
df['is_wicket'] = df['wicket_kind'].notna() & ~df['wicket_kind'].isin(NON_BOWLER_WICKETS)

# ───────────────────────────────────────────────────────
# STEP 4 — BUILD MATCH-LEVEL DATAFRAME (1 row per match)
# WHY: Toss analysis needs match-level data, not ball-level
# ───────────────────────────────────────────────────────
matches = df.groupby('match_id').first()[
    ['season', 'toss_winner', 'toss_decision', 'winner']
].reset_index()

# 1 if toss winner also won the match, else 0
matches['toss_won_match'] = (matches['toss_winner'] == matches['winner']).astype(int)

print(f"Match-level rows: {len(matches)}")

# ───────────────────────────────────────────────────────
# QUESTION 1 — Does winning the toss lead to winning?
# ───────────────────────────────────────────────────────
toss_win_pct  = matches['toss_won_match'].mean() * 100
toss_lose_pct = 100 - toss_win_pct

print(f"\nQ1 Results:")
print(f"  Toss winners win rate : {toss_win_pct:.1f}%")
print(f"  Toss losers  win rate : {toss_lose_pct:.1f}%")

# Breakdown by toss decision (bat vs field)
decision_stats = matches.groupby('toss_decision')['toss_won_match'].agg(['mean','count'])
decision_stats['win_pct'] = decision_stats['mean'] * 100
print(f"\n  By decision:")
print(decision_stats[['win_pct','count']].to_string())

# ───────────────────────────────────────────────────────
# QUESTION 2 — Which phase is most linked to winning?
# ───────────────────────────────────────────────────────

# Group by match + batting team + phase → total runs in that phase
phase_df = (df.groupby(['match_id', 'batting_team', 'phase'])['runs_total']
              .sum().reset_index())

# Join match winner info
phase_df = phase_df.merge(matches[['match_id', 'winner']], on='match_id')

# Tag each row: did this batting team win the match?
phase_df['result'] = (phase_df['batting_team'] == phase_df['winner']).map(
    {True: 'Winning Teams', False: 'Losing Teams'})

# Average runs per phase for winners vs losers
phase_avg = phase_df.groupby(['phase', 'result'])['runs_total'].mean().unstack()

phase_order = ['Powerplay\n(Overs 1–6)', 'Middle Overs\n(Overs 7–15)', 'Death Overs\n(Overs 16–20)']
phase_avg = phase_avg.reindex(phase_order)

print(f"\nQ2 Results — Avg runs per phase:")
print(phase_avg.to_string())

# ───────────────────────────────────────────────────────
# QUESTION 3 — Top 5 Batters & Bowlers
# ───────────────────────────────────────────────────────

# Batters: use runs_batter only (not extras, not team extras)
batter_stats = (df.groupby('batter')
                  .agg(total_runs=('runs_batter', 'sum'),
                       innings=('match_id', lambda x: x.nunique()))
                  .reset_index()
                  .sort_values('total_runs', ascending=False)
                  .head(5))
batter_stats['rank'] = range(1, 6)

# Bowlers: credit only legitimate bowling wickets
bowler_stats = (df[df['is_wicket']]
                  .groupby('bowler')
                  .agg(total_wickets=('is_wicket', 'sum'),
                       matches=('match_id', lambda x: x.nunique()))
                  .reset_index()
                  .sort_values('total_wickets', ascending=False)
                  .head(5))
bowler_stats['rank'] = range(1, 6)

print(f"\nQ3 — Top 5 Batters:")
print(batter_stats[['rank', 'batter', 'total_runs', 'innings']].to_string(index=False))
print(f"\nQ3 — Top 5 Bowlers:")
print(bowler_stats[['rank', 'bowler', 'total_wickets', 'matches']].to_string(index=False))

# ───────────────────────────────────────────────────────
# CHART STYLING CONSTANTS
# ───────────────────────────────────────────────────────
BG       = '#0d1117'
CARD     = '#161b22'
WIN_C    = '#00c896'
LOSE_C   = '#ff6b6b'
ACCENT   = '#f5a623'
TEXT_PRI = '#e6edf3'
TEXT_SEC = '#8b949e'
GRID_C   = '#21262d'

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'text.color':       TEXT_PRI,
    'axes.facecolor':   CARD,
    'figure.facecolor': BG,
    'axes.edgecolor':   GRID_C,
    'axes.labelcolor':  TEXT_SEC,
    'xtick.color':      TEXT_SEC,
    'ytick.color':      TEXT_SEC,
    'grid.color':       GRID_C,
    'grid.linewidth':   0.6,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ───────────────────────────────────────────────────────
# FIGURE LAYOUT
# ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 2, figure=fig,
                         height_ratios=[0.22, 1, 1.5, 0.25],
                         hspace=0.55, wspace=0.35,
                         top=0.97, bottom=0.02, left=0.06, right=0.97)

# ── Title row ──
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_facecolor(BG)
for sp in ax_title.spines.values(): sp.set_visible(False)
ax_title.set_xticks([]); ax_title.set_yticks([])
ax_title.text(0.5, 0.85, '🏏  IPL Data Analysis  ·  2019 – 2023',
              transform=ax_title.transAxes, ha='center', va='top',
              fontsize=22, fontweight='bold', color=TEXT_PRI)
ax_title.text(0.5, 0.32,
              f'5 Seasons  ·  {matches["match_id"].nunique()} Matches  ·  '
              f'{len(df):,} Balls  ·  {df["is_wicket"].sum():,} Wickets',
              transform=ax_title.transAxes, ha='center', va='top',
              fontsize=11.5, color=TEXT_SEC)

# ───────────────────────────────────────────────────────
# CHART 1 — Toss Win Rate (left column)
# ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
labels = ['Toss Winners', 'Toss Losers']
values = [toss_win_pct, toss_lose_pct]
x = np.arange(len(labels))

bars = ax1.bar(x, values, width=0.45, color=[WIN_C, LOSE_C], zorder=3, edgecolor='none')

# 50% reference line — the "coin flip" baseline
ax1.axhline(50, color=ACCENT, linestyle='--', linewidth=1.4, alpha=0.7, zorder=2)

for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
             f'{val:.1f}%', ha='center', va='bottom',
             fontsize=13, fontweight='bold', color=TEXT_PRI)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11.5)
ax1.set_ylabel('Win Percentage (%)', fontsize=10.5)
ax1.set_ylim(0, 70)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
ax1.grid(axis='y', zorder=0)
ax1.set_title('Q1 · Does Winning the Toss Mean Winning the Match?',
              fontsize=12, fontweight='bold', color=TEXT_PRI, pad=12)

legend_patch = mpatches.Patch(color=ACCENT, label='50% baseline (random)')
ax1.legend(handles=[legend_patch], loc='upper right',
           facecolor=CARD, edgecolor=GRID_C, fontsize=9, labelcolor=TEXT_SEC)

# ───────────────────────────────────────────────────────
# CHART 2 — Phase Analysis (right column)
# ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])

win_runs  = [phase_avg.loc[p, 'Winning Teams']  for p in phase_order]
lose_runs = [phase_avg.loc[p, 'Losing Teams']   for p in phase_order]
x2 = np.arange(len(phase_order))

b_win  = ax2.bar(x2 - 0.175, win_runs,  width=0.35, color=WIN_C,  label='Winning Teams', zorder=3, edgecolor='none')
b_lose = ax2.bar(x2 + 0.175, lose_runs, width=0.35, color=LOSE_C, label='Losing Teams',  zorder=3, edgecolor='none')

for bar, val in zip(b_win, win_runs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
             f'{val:.0f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color=WIN_C)
for bar, val in zip(b_lose, lose_runs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
             f'{val:.0f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color=LOSE_C)

# Show gap between winner and loser above each pair
for i, (w, l) in enumerate(zip(win_runs, lose_runs)):
    ax2.annotate(f'+{w-l:.0f}', xy=(x2[i], max(w, l) + 4),
                 ha='center', fontsize=9, color=ACCENT, fontweight='bold')

ax2.set_xticks(x2)
ax2.set_xticklabels(phase_order, fontsize=10)
ax2.set_ylabel('Avg Runs per Innings Phase', fontsize=10.5)
ax2.set_ylim(0, 105)
ax2.grid(axis='y', zorder=0)
ax2.set_title('Q2 · Which Phase Separates Winners from Losers?',
              fontsize=12, fontweight='bold', color=TEXT_PRI, pad=12)
ax2.legend(facecolor=CARD, edgecolor=GRID_C, fontsize=9.5,
           labelcolor=TEXT_SEC, loc='upper left')

# ───────────────────────────────────────────────────────
# TABLE — Top 5 Batters & Bowlers
# ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :])
ax3.set_facecolor(CARD)
for sp in ax3.spines.values(): sp.set_visible(False)
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_title('Q3 · Top 5 Batters & Top 5 Bowlers  (2019 – 2023)',
              fontsize=13, fontweight='bold', color=TEXT_PRI, pad=14)

bat_data  = [[str(r['rank']), r['batter'], f"{r['total_runs']:,}", str(r['innings'])]
             for _, r in batter_stats.iterrows()]
bowl_data = [[str(r['rank']), r['bowler'], str(r['total_wickets']), str(r['matches'])]
             for _, r in bowler_stats.iterrows()]

def draw_table(ax, data, col_labels, x_start, col_widths, title_text, header_color):
    total_w = sum(col_widths)
    header_h = 0.13; row_h = 0.13; y_start = 0.88

    ax.text(x_start + total_w/2, y_start + 0.07, title_text,
            ha='center', va='center', fontsize=11.5, fontweight='bold',
            color=header_color, transform=ax.transAxes)

    x = x_start
    for col, w in zip(col_labels, col_widths):
        rect = mpatches.FancyBboxPatch((x, y_start - header_h), w - 0.005, header_h,
                                        boxstyle='round,pad=0.005', facecolor=header_color,
                                        edgecolor='none', transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y_start - header_h/2, col, ha='center', va='center',
                fontsize=10, fontweight='bold', color=BG, transform=ax.transAxes, zorder=3)
        x += w

    for row_i, row in enumerate(data):
        y = y_start - header_h - (row_i + 1) * row_h
        bg = '#1c2128' if row_i % 2 == 0 else CARD
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_start, y), total_w - 0.005, row_h - 0.005,
            boxstyle='round,pad=0.002', facecolor=bg,
            edgecolor=GRID_C, linewidth=0.4, transform=ax.transAxes, zorder=1))
        x = x_start
        for j, (cell, w) in enumerate(zip(row, col_widths)):
            c = ACCENT if j == 0 else (header_color if j == 2 else TEXT_PRI)
            fw = 'bold' if j in (0, 2) else 'normal'
            ax.text(x + w/2, y + row_h/2, cell, ha='center', va='center',
                    fontsize=10.5, color=c, fontweight=fw, transform=ax.transAxes, zorder=3)
            x += w

draw_table(ax3, bat_data,  ['#','Batter', 'Runs',   'Innings'],
           x_start=0.03, col_widths=[0.07,0.24,0.10,0.10],
           title_text='🏏  Top 5 Batters by Runs',   header_color=WIN_C)

draw_table(ax3, bowl_data, ['#','Bowler', 'Wickets','Matches'],
           x_start=0.53, col_widths=[0.07,0.24,0.10,0.10],
           title_text='🎯  Top 5 Bowlers by Wickets', header_color='#7c9ef5')

ax3.set_xlim(0,1); ax3.set_ylim(0,1)

# ───────────────────────────────────────────────────────
# SURPRISE INSIGHT
# ───────────────────────────────────────────────────────
mid_gap = (phase_avg.loc['Middle Overs\n(Overs 7–15)','Winning Teams'] -
           phase_avg.loc['Middle Overs\n(Overs 7–15)','Losing Teams'])

ax4 = fig.add_subplot(gs[3, :])
ax4.set_facecolor('#1a1f29')
for sp in ax4.spines.values(): sp.set_visible(False)
ax4.set_xticks([]); ax4.set_yticks([])
ax4.text(0.5, 0.55,
         f'💡  Surprise: Toss barely matters ({toss_win_pct:.0f}% win rate) — but winning teams score '
         f'{mid_gap:.0f} more runs in Middle Overs, showing that the "boring phase" is where IPL matches are decided.',
         transform=ax4.transAxes, ha='center', va='center',
         fontsize=11, color=ACCENT, fontstyle='italic')

# ───────────────────────────────────────────────────────
# SAVE
# ───────────────────────────────────────────────────────
OUTPUT_FILE = 'ipl_eda_dashboard.png'   # ← output path
fig.savefig(OUTPUT_FILE, dpi=160, bbox_inches='tight', facecolor=BG)
print(f"Dashboard saved to: {OUTPUT_FILE}")
plt.close()