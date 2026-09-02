using System;
using System.Threading.Tasks;
using Xaviris.Core.Profiling;

namespace Xaviris.Core.Examples
{
    public static class ProfilerUsageExample
    {
        public static async Task RunProfilerExample()
        {
            Console.WriteLine("=== System Profiler Example ===\n");

            // Create profiler with 500ms sampling interval, retaining last 1000 samples
            using var profiler = new SystemProfiler(samplingIntervalMs: 500, maxSnapshots: 1000);

            Console.WriteLine("Starting background profiler...");
            profiler.Start();

            // Let it run for 10 seconds while your application does work
            Console.WriteLine("Profiling for 10 seconds...\n");
            await Task.Delay(10000);

            // Get recent snapshots (last 10 samples)
            var recentSnapshots = profiler.GetRecentSnapshots(10);
            Console.WriteLine($"Recent Metrics (last {recentSnapshots.Count} samples):");
            foreach (var snapshot in recentSnapshots)
            {
                Console.WriteLine($"  {snapshot}");
            }

            // Get statistics
            var stats = profiler.GetStatistics();
            Console.WriteLine($"\nAggregated Statistics:\n  {stats}\n");

            // Get a single on-demand snapshot
            var current = profiler.GetCurrentSnapshot();
            Console.WriteLine($"Current Snapshot:\n  {current}\n");

            // Stop profiling
            Console.WriteLine("Stopping profiler...");
            await profiler.StopAsync();
            Console.WriteLine("Done.");
        }

        public static async Task RunContinuousMonitoring(int durationSeconds = 30)
        {
            Console.WriteLine("=== Continuous Monitoring Example ===\n");

            using var profiler = new SystemProfiler(samplingIntervalMs: 500, maxSnapshots: 200);
            profiler.Start();

            var loggingTask = Task.Run(async () =>
            {
                int elapsedSeconds = 0;
                while (elapsedSeconds < durationSeconds)
                {
                    await Task.Delay(5000); // Log every 5 seconds
                    var stats = profiler.GetStatistics();
                    Console.WriteLine($"[{elapsedSeconds + 5}s] {stats}");
                    elapsedSeconds += 5;
                }
            });

            // Simulate application work
            await Task.Delay(TimeSpan.FromSeconds(durationSeconds));

            await profiler.StopAsync();
            await loggingTask;

            // Final report
            var finalStats = profiler.GetStatistics();
            Console.WriteLine($"\nFinal Report:\n  {finalStats}");
        }

        public static async Task ProfileInferenceWorkload()
        {
            Console.WriteLine("=== Inference Workload Profiling ===\n");

            using var profiler = new SystemProfiler(samplingIntervalMs: 250); // Faster sampling for detailed profile
            profiler.Start();

            Console.WriteLine("Running inference workload...");
            // Your inference/processing code here
            await Task.Delay(5000);

            await profiler.StopAsync();

            var stats = profiler.GetStatistics();
            Console.WriteLine($"\nWorkload Performance Metrics:");
            Console.WriteLine($"  Total Samples: {stats.SampleCount}");
            Console.WriteLine($"  CPU Usage: {stats.AvgCpuPercent:F1}% (range: {stats.MinCpuPercent:F1}% - {stats.MaxCpuPercent:F1}%)");
            Console.WriteLine($"  Memory: {stats.AvgMemoryMB:F1} MB (range: {stats.MinMemoryMB:F1} - {stats.MaxMemoryMB:F1})");
            Console.WriteLine($"  Thread Count: {stats.AvgThreadCount}");
        }
    }
}
