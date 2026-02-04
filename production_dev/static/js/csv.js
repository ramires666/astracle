/**
 * CSV Export Helper
 */

export function exportForecastToCSV(predictions) {
    if (!predictions || predictions.length === 0) {
        alert('No predictions to export. Generate a forecast first.');
        return;
    }

    const headers = ['Date', 'Direction', 'Confidence', 'Simulated Price'];
    const rows = predictions.map((pred) => [
        pred.date,
        pred.direction,
        Number(pred.confidence ?? 0.5).toFixed(4),
        Number(pred.simulated_price ?? 0).toFixed(2),
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `btc_astro_forecast_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);
}

