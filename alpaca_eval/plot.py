import matplotlib.pyplot as plt

length_controled_win_rate = {
    'full_template': {
        'pass@1': 2.508000579093168,
        'pass@4': 2.916727187892163,
        'pass@8': 3.187975572155956,
        'pass@16': 3.52838159189206,
        'pass@32': 3.900254278896008,
    },
    'simple_steer': {
        'pass@1': 0.9093423902444938,
        'pass@4': 1.291576775985285,
        'pass@8': 1.4474126248191603,
        'pass@16': 1.892368941119056,
        'pass@32': 2.4115446378692975,
    },
    'minimum_dialog': {
        'pass@1': 2.3973653594997404,
        'pass@4': 3.1792813790008854,
        'pass@8': 3.636269073849325,
        'pass@16': 4.432365118435249,
        'pass@32': 4.799251580097123
    }
}

font_size = 20
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(10, 6))

palette = ['#012a4a', '#013a63', '#01497c', '#014f86', '#468faf', '#61a5c2', '#89c2d9', '#a9d6e5']
plt.plot(length_controled_win_rate['full_template'].keys(), length_controled_win_rate['full_template'].values(), label='full_template', linewidth=3, color=palette[0], marker='o', markersize=10)
plt.plot(length_controled_win_rate['simple_steer'].keys(), length_controled_win_rate['simple_steer'].values(), label='simple_steer', linewidth=3, color=palette[3], marker='o', markersize=10)
plt.plot(length_controled_win_rate['minimum_dialog'].keys(), length_controled_win_rate['minimum_dialog'].values(), label='minimum_dialog', linewidth=3, color=palette[6], marker='o', markersize=10)

plt.ylabel('Win Rate', fontsize=font_size)
ticks_font_size = font_size - 2
plt.xticks(fontsize=ticks_font_size)
plt.yticks(fontsize=ticks_font_size)
plt.legend(fontsize=font_size)
plt.savefig('alpaca_eval/figs/length_controled_win_rate.pdf', dpi=300, bbox_inches='tight')

