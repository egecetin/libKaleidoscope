a = 2492;
b = -2.165e-6;
c = 364.9;
d = -2.08e-7;
xPts = 1:10e6;

x = [720*576 1280*720 1920*1080 3840*2160];
y = [1350 640 265 65];

figure;
scatter(x, y, 'r*', 'LineWidth', 1);
hold on;
plot(xPts,a*exp(b*xPts)+c*exp(d*xPts), 'b--', 'LineWidth', 2);

for idx=1:length(y)
    plot([x(idx) x(idx)], [0 y(idx)], 'r--', 'LineWidth', 2);
end

legend({'Measurements', 'Estimation'});
xlabel('Resolution', 'FontWeight', 'bold');
ylabel('FPS', 'FontWeight', 'bold');
title('Performance');
xticks([720*576 1280*720 1920*1080 3840*2160 7680*4320])
xticklabels({'576p', '720p', '1080p', '4K UHD', '8K UHD'})
yticks([60 120 240 480 960 1920])
yticklabels({'60', '120', '240', '480', '960', '1920'})
set(gca,'linewidth',2)
set(get(gca, 'XAxis'), 'FontWeight', 'bold');
set(get(gca, 'YAxis'), 'FontWeight', 'bold');

saveas(gcf, 'performance.png');