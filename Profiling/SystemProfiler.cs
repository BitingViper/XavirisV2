using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace Xaviris.Core.Profiling
{
    public readonly struct SystemMetricsSnapshot
    {
        /// <summary>Timestamp when metrics were sampled.</summary>
        public readonly DateTime Timestamp;

        /// <summary>CPU usage percentage (0-100).</summary>
        public readonly float CpuUsagePercent;

        /// <summary>Working set memory in MB.</summary>
        public readonly float MemoryMB;

        /// <summary>Memory usage percentage based on available system memory.</summary>
        public readonly float MemoryPercent;

        /// <summary>Thread count in the current process.</summary>
        public readonly int ThreadCount;

        /// <summary>Garbage collection statistics (Gen 0, 1, 2 collections).</summary>
        public readonly GCMetrics GCStats;

        /// <summary>Total elapsed milliseconds since process start.</summary>
        public readonly double ElapsedMilliseconds;

        public SystemMetricsSnapshot(
            DateTime timestamp,
            float cpuUsage,
            float memoryMB,
            float memoryPercent,
            int threadCount,
            GCMetrics gcStats,
            double elapsed)
        {
            Timestamp = timestamp;
            CpuUsagePercent = cpuUsage;
            MemoryMB = memoryMB;
            MemoryPercent = memoryPercent;
            ThreadCount = threadCount;
            GCStats = gcStats;
            ElapsedMilliseconds = elapsed;
        }

        public override string ToString()
        {
            return $"[{Timestamp:yyyy-MM-dd HH:mm:ss.fff}] CPU: {CpuUsagePercent:F1}% | Memory: {MemoryMB:F1} MB ({MemoryPercent:F1}%) | Threads: {ThreadCount} | GC: Gen0={GCStats.Gen0Collections} Gen1={GCStats.Gen1Collections} Gen2={GCStats.Gen2Collections}";
        }
    }

    public readonly struct GCMetrics
    {
        public readonly long Gen0Collections;
        public readonly long Gen1Collections;
        public readonly long Gen2Collections;
        public readonly long TotalMemoryBytes;

        public GCMetrics(long gen0, long gen1, long gen2, long totalMemory)
        {
            Gen0Collections = gen0;
            Gen1Collections = gen1;
            Gen2Collections = gen2;
            TotalMemoryBytes = totalMemory;
        }
    }

    public class SystemProfiler : IDisposable
    {
        private readonly int _samplingIntervalMs;
        private readonly int _maxSnapshots;
        private readonly Process _currentProcess;
        private readonly DateTime _processStartTime;
        private PerformanceCounter? _cpuCounter;
        private CancellationTokenSource? _cts;
        private Task? _profilingTask;
        private readonly object _lock = new object();
        private readonly Queue<SystemMetricsSnapshot> _snapshots;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the SystemProfiler.
        /// </summary>
        /// <param name="samplingIntervalMs">Interval between samples in milliseconds (default 500).</param>
        /// <param name="maxSnapshots">Maximum snapshots to retain in memory (default 1000, ~500 seconds).</param>
        public SystemProfiler(int samplingIntervalMs = 500, int maxSnapshots = 1000)
        {
            if (samplingIntervalMs < 100)
                throw new ArgumentException("Sampling interval must be >= 100ms", nameof(samplingIntervalMs));

            _samplingIntervalMs = samplingIntervalMs;
            _maxSnapshots = maxSnapshots;
            _currentProcess = Process.GetCurrentProcess();
            _processStartTime = DateTime.UtcNow;
            _snapshots = new Queue<SystemMetricsSnapshot>(maxSnapshots);

            // Initialize CPU performance counter (Windows-only, with graceful fallback)
            try
            {
#pragma warning disable CA1416
                _cpuCounter = new PerformanceCounter(
                    "Processor",
                    "% Processor Time",
                    "_Total",
                    true);
                _ = _cpuCounter.NextValue();
#pragma warning restore CA1416
            }
            catch
            {
                _cpuCounter = null;
            }

            _cts = new CancellationTokenSource();
            _disposed = false;
        }

        public void Start()
        {
            lock (_lock)
            {
                if (_profilingTask != null && !_profilingTask.IsCompleted)
                    return; // Already running

                _cts = new CancellationTokenSource();
                _profilingTask = Task.Factory.StartNew(
                    () => ProfilerLoop(_cts.Token),
                    CancellationToken.None,
                    TaskCreationOptions.LongRunning,
                    TaskScheduler.Default);
            }
        }

        public async Task StopAsync()
        {
            lock (_lock)
            {
                if (_profilingTask == null || _profilingTask.IsCompleted)
                    return;

                _cts?.Cancel();
            }

            try
            {
                await _profilingTask.ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Expected when cancelling
            }
        }

        public SystemMetricsSnapshot GetCurrentSnapshot()
        {
            return SampleMetrics();
        }

        public List<SystemMetricsSnapshot> GetSnapshots()
        {
            lock (_lock)
            {
                return new List<SystemMetricsSnapshot>(_snapshots);
            }
        }

        public List<SystemMetricsSnapshot> GetRecentSnapshots(int count)
        {
            lock (_lock)
            {
                var result = new List<SystemMetricsSnapshot>(Math.Min(count, _snapshots.Count));
                int startIdx = Math.Max(0, _snapshots.Count - count);
                int idx = 0;

                foreach (var snapshot in _snapshots)
                {
                    if (idx++ >= startIdx)
                        result.Add(snapshot);
                }

                return result;
            }
        }

        public void ClearSnapshots()
        {
            lock (_lock)
            {
                _snapshots.Clear();
            }
        }

        public ProfilerStatistics GetStatistics()
        {
            lock (_lock)
            {
                if (_snapshots.Count == 0)
                    return new ProfilerStatistics();

                float avgCpu = 0, maxCpu = 0, minCpu = float.MaxValue;
                float avgMemory = 0, maxMemory = 0, minMemory = float.MaxValue;
                int avgThreads = 0;

                foreach (var snapshot in _snapshots)
                {
                    avgCpu += snapshot.CpuUsagePercent;
                    maxCpu = Math.Max(maxCpu, snapshot.CpuUsagePercent);
                    minCpu = Math.Min(minCpu, snapshot.CpuUsagePercent);

                    avgMemory += snapshot.MemoryMB;
                    maxMemory = Math.Max(maxMemory, snapshot.MemoryMB);
                    minMemory = Math.Min(minMemory, snapshot.MemoryMB);

                    avgThreads += snapshot.ThreadCount;
                }

                int count = _snapshots.Count;
                return new ProfilerStatistics(
                    count,
                    avgCpu / count,
                    minCpu,
                    maxCpu,
                    avgMemory / count,
                    minMemory,
                    maxMemory,
                    avgThreads / count);
            }
        }

        private void ProfilerLoop(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    var snapshot = SampleMetrics();

                    lock (_lock)
                    {
                        _snapshots.Enqueue(snapshot);
                        while (_snapshots.Count > _maxSnapshots)
                            _snapshots.Dequeue();
                    }

                    Task.Delay(_samplingIntervalMs, cancellationToken).Wait(cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[Profiler] Error in profiling loop: {ex.Message}");
                }
            }
        }

        private SystemMetricsSnapshot SampleMetrics()
        {
            _currentProcess.Refresh();

#pragma warning disable CA1416
            float cpuUsage = _cpuCounter?.NextValue() ?? 0;
#pragma warning restore CA1416
            float memoryMB = _currentProcess.WorkingSet64 / (1024f * 1024f);
            long totalMemory = GC.GetTotalMemory(false);
            float memoryPercent = GC.GetTotalMemory(false) / (1024f * 1024f * 1024f) * 100; // Approximate based on 1GB reference

            var gcStats = new GCMetrics(
                GC.CollectionCount(0),
                GC.CollectionCount(1),
                GC.CollectionCount(2),
                totalMemory);

            double elapsedMs = (DateTime.UtcNow - _processStartTime).TotalMilliseconds;

            return new SystemMetricsSnapshot(
                DateTime.UtcNow,
                cpuUsage,
                memoryMB,
                memoryPercent,
                _currentProcess.Threads.Count,
                gcStats,
                elapsedMs);
        }

        public void Dispose()
        {
            if (_disposed) return;

            _cts?.Cancel();
            try
            {
                _profilingTask?.Wait(TimeSpan.FromSeconds(5));
            }
            catch { }

            _cts?.Dispose();
            _cpuCounter?.Dispose();
            _currentProcess?.Dispose();

            _disposed = true;
        }
    }

    public readonly struct ProfilerStatistics
    {
        public readonly int SampleCount;
        public readonly float AvgCpuPercent;
        public readonly float MinCpuPercent;
        public readonly float MaxCpuPercent;
        public readonly float AvgMemoryMB;
        public readonly float MinMemoryMB;
        public readonly float MaxMemoryMB;
        public readonly int AvgThreadCount;

        public ProfilerStatistics(
            int count,
            float avgCpu,
            float minCpu,
            float maxCpu,
            float avgMemory,
            float minMemory,
            float maxMemory,
            int avgThreads)
        {
            SampleCount = count;
            AvgCpuPercent = avgCpu;
            MinCpuPercent = minCpu;
            MaxCpuPercent = maxCpu;
            AvgMemoryMB = avgMemory;
            MinMemoryMB = minMemory;
            MaxMemoryMB = maxMemory;
            AvgThreadCount = avgThreads;
        }

        public override string ToString()
        {
            return $"Samples: {SampleCount} | CPU: {AvgCpuPercent:F1}% (min: {MinCpuPercent:F1}%, max: {MaxCpuPercent:F1}%) | Memory: {AvgMemoryMB:F1} MB (min: {MinMemoryMB:F1}, max: {MaxMemoryMB:F1}) | Threads: {AvgThreadCount}";
        }
    }
}
