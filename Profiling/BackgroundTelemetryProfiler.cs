using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace Xaviris.Core.Profiling
{
    public sealed class VideoFramePacket
    {
        public long PacketId { get; init; }
        public int StreamId { get; init; }
        public int FrameNumber { get; init; }
        public DateTime TimestampUtc { get; init; }
        public byte[] RawImage { get; init; } = Array.Empty<byte>();
        public RectangleF[] BoundingBoxes { get; init; } = Array.Empty<RectangleF>();
        public string[] Labels { get; init; } = Array.Empty<string>();
        public float Fps { get; init; }
        public float ProcessingLatencyMs { get; init; }
        public float PacketDropRate { get; init; }
        public float CpuCore0 { get; init; }
        public float CpuCore1 { get; init; }
        public float CpuCore2 { get; init; }
        public float CpuCore3 { get; init; }
        public float TotalCpuPercent { get; init; }
        public float MemoryMb { get; init; }
        public int ThreadCount { get; init; }
    }

    public sealed class StreamFrameMetrics
    {
        public int StreamId { get; set; }
        public float Fps { get; set; }
        public float LatencyMs { get; set; }
        public float PacketDropRate { get; set; }
        public int FramesProcessed { get; set; }
        public long LastFrameTimestampUtc { get; set; }

        public void Update(float fps, float latencyMs, float dropRate)
        {
            Fps = fps;
            LatencyMs = latencyMs;
            PacketDropRate = dropRate;
            FramesProcessed++;
            LastFrameTimestampUtc = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        }
    }

    public sealed class BackgroundTelemetryProfiler : IDisposable
    {
        private readonly int _samplingIntervalMs;
        private readonly string _csvPath;
        private readonly Channel<VideoFramePacket> _packetQueue;
        private readonly ConcurrentDictionary<int, StreamFrameMetrics> _streamStats;
        private readonly object _csvLock = new();
        private readonly SystemProfiler _systemProfiler;
        private readonly CancellationTokenSource _cts = new();
        private readonly Task _exportTask;
        private readonly PerformanceCounter[] _cpuCoreCounters;
        private readonly long _processStartUtcTicks;
        private readonly string[] _csvHeader;
        private long _sequence;
        private bool _disposed;

        public event Action<VideoFramePacket>? PacketPublished;

        public ChannelReader<VideoFramePacket> PacketReader => _packetQueue.Reader;

        public BackgroundTelemetryProfiler(
            string? csvPath = null,
            int samplingIntervalMs = 500,
            int maxBufferedPackets = 2048)
        {
            if (samplingIntervalMs <= 0)
                throw new ArgumentOutOfRangeException(nameof(samplingIntervalMs));

            _samplingIntervalMs = samplingIntervalMs;
            _csvPath = string.IsNullOrWhiteSpace(csvPath)
                ? Path.Combine(AppContext.BaseDirectory, $"xaviris_telemetry_{DateTime.UtcNow:yyyyMMdd_HHmmss}.csv")
                : csvPath;

            _packetQueue = Channel.CreateBounded<VideoFramePacket>(new BoundedChannelOptions(maxBufferedPackets)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = false
            });

            _streamStats = new ConcurrentDictionary<int, StreamFrameMetrics>();
            _systemProfiler = new SystemProfiler(samplingIntervalMs: samplingIntervalMs, maxSnapshots: 1000);
            _processStartUtcTicks = DateTime.UtcNow.Ticks;

            _csvHeader = new[]
            {
                "Timestamp_UTC",
                "Stream_ID",
                "Frame_Number",
                "FPS",
                "Latency_ms",
                "Packet_Drop_Rate",
                "CPU_Core_0_pct",
                "CPU_Core_1_pct",
                "CPU_Core_2_pct",
                "CPU_Core_3_pct",
                "CPU_Total_pct",
                "Memory_MB",
                "Thread_Count",
                "GC_Gen0_Collections",
                "GC_Gen1_Collections",
                "GC_Gen2_Collections",
                "Detection_Count",
                "Image_Bytes",
                "Packet_ID"
            };

            _cpuCoreCounters = CreateCoreCounters();
            _systemProfiler.Start();
            _exportTask = Task.Factory.StartNew(
                () => CsvExportLoop(_cts.Token),
                CancellationToken.None,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);

            EnsureCsvFile();
            Console.WriteLine($"[Telemetry] CSV output: {_csvPath}");
        }

        public void RegisterStream(int streamId)
        {
            _streamStats.TryAdd(streamId, new StreamFrameMetrics { StreamId = streamId });
        }

        public void UpdateStreamTelemetry(int streamId, float fps, float latencyMs, float packetDropRate)
        {
            _streamStats.AddOrUpdate(
                streamId,
                _ => new StreamFrameMetrics { StreamId = streamId, Fps = fps, LatencyMs = latencyMs, PacketDropRate = packetDropRate },
                (_, existing) =>
                {
                    existing.Update(fps, latencyMs, packetDropRate);
                    return existing;
                });
        }

        public VideoFramePacket CreateFramePacket(
            int streamId,
            int frameNumber,
            byte[] rawImage,
            RectangleF[] boundingBoxes,
            string[] labels,
            float? fps = null,
            float? latencyMs = null,
            float? packetDropRate = null)
        {
            var coreUsage = SampleCpuCoreUsage();
            var sysSnapshot = _systemProfiler.GetCurrentSnapshot();
            var frameTelemetry = _streamStats.TryGetValue(streamId, out var stats)
                ? new StreamFrameMetrics { StreamId = streamId, Fps = stats.Fps, LatencyMs = stats.LatencyMs, PacketDropRate = stats.PacketDropRate }
                : new StreamFrameMetrics { StreamId = streamId };

            return new VideoFramePacket
            {
                PacketId = Interlocked.Increment(ref _sequence),
                StreamId = streamId,
                FrameNumber = frameNumber,
                TimestampUtc = DateTime.UtcNow,
                RawImage = rawImage ?? Array.Empty<byte>(),
                BoundingBoxes = boundingBoxes ?? Array.Empty<RectangleF>(),
                Labels = labels ?? Array.Empty<string>(),
                Fps = fps ?? frameTelemetry.Fps,
                ProcessingLatencyMs = latencyMs ?? frameTelemetry.LatencyMs,
                PacketDropRate = packetDropRate ?? frameTelemetry.PacketDropRate,
                CpuCore0 = coreUsage[0],
                CpuCore1 = coreUsage[1],
                CpuCore2 = coreUsage[2],
                CpuCore3 = coreUsage[3],
                TotalCpuPercent = sysSnapshot.CpuUsagePercent,
                MemoryMb = sysSnapshot.MemoryMB,
                ThreadCount = sysSnapshot.ThreadCount
            };
        }

        public async ValueTask PublishFrameAsync(VideoFramePacket packet, CancellationToken cancellationToken = default)
        {
            await _packetQueue.Writer.WriteAsync(packet, cancellationToken).ConfigureAwait(false);
            PacketPublished?.Invoke(packet);
        }

        public void WriteTelemetryRow(VideoFramePacket packet)
        {
            var row = new StringBuilder();
            row.Append(packet.TimestampUtc.ToString("O")).Append(',');
            row.Append(packet.StreamId).Append(',');
            row.Append(packet.FrameNumber).Append(',');
            row.Append(packet.Fps.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).Append(',');
            row.Append(packet.ProcessingLatencyMs.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).Append(',');
            row.Append(packet.PacketDropRate.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).Append(',');
            row.Append(packet.CpuCore0.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).Append(',');
            row.Append(packet.CpuCore1.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).Append(',');
            row.Append(packet.CpuCore2.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).Append(',');
            row.Append(packet.CpuCore3.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).Append(',');
            row.Append(packet.TotalCpuPercent.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).Append(',');
            row.Append(packet.MemoryMb.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).Append(',');
            row.Append(packet.ThreadCount).Append(',');
            row.Append(GC.CollectionCount(0)).Append(',');
            row.Append(GC.CollectionCount(1)).Append(',');
            row.Append(GC.CollectionCount(2)).Append(',');
            row.Append(packet.BoundingBoxes.Length).Append(',');
            row.Append(packet.RawImage.Length).Append(',');
            row.Append(packet.PacketId);

            lock (_csvLock)
            {
                File.AppendAllText(_csvPath, row.ToString() + Environment.NewLine);
            }
        }

        public void Stop()
        {
            if (_disposed)
                return;

            _disposed = true;
            _cts.Cancel();
            _systemProfiler.StopAsync().GetAwaiter().GetResult();
            _packetQueue.Writer.TryComplete();
            _exportTask.GetAwaiter().GetResult();
        }

        private void EnsureCsvFile()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_csvPath)!);
            if (!File.Exists(_csvPath))
            {
                lock (_csvLock)
                {
                    File.WriteAllText(_csvPath, string.Join(',', _csvHeader) + Environment.NewLine);
                }
            }
        }

        private async Task CsvExportLoop(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(_samplingIntervalMs, cancellationToken).ConfigureAwait(false);

                    var cpu = SampleCpuCoreUsage();
                    var snapshot = _systemProfiler.GetCurrentSnapshot();
                    var streamRows = new List<VideoFramePacket>();

                    foreach (var kvp in _streamStats)
                    {
                        streamRows.Add(new VideoFramePacket
                        {
                            PacketId = Interlocked.Increment(ref _sequence),
                            StreamId = kvp.Key,
                            FrameNumber = 0,
                            TimestampUtc = DateTime.UtcNow,
                            RawImage = Array.Empty<byte>(),
                            BoundingBoxes = Array.Empty<RectangleF>(),
                            Labels = Array.Empty<string>(),
                            Fps = kvp.Value.Fps,
                            ProcessingLatencyMs = kvp.Value.LatencyMs,
                            PacketDropRate = kvp.Value.PacketDropRate,
                            CpuCore0 = cpu[0],
                            CpuCore1 = cpu[1],
                            CpuCore2 = cpu[2],
                            CpuCore3 = cpu[3],
                            TotalCpuPercent = snapshot.CpuUsagePercent,
                            MemoryMb = snapshot.MemoryMB,
                            ThreadCount = snapshot.ThreadCount
                        });
                    }

                    foreach (var row in streamRows)
                    {
                        WriteTelemetryRow(row);
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[Telemetry] CSV export error: {ex.Message}");
                }
            }
        }

        private static PerformanceCounter[] CreateCoreCounters()
        {
#pragma warning disable CA1416
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return Array.Empty<PerformanceCounter>();
            }

            var counters = new List<PerformanceCounter>();

            try
            {
                for (int i = 0; i < 4; i++)
                {
                    var counter = new PerformanceCounter("Processor Information", "% Processor Time", i.ToString(), true);
                    _ = counter.NextValue();
                    counters.Add(counter);
                }
            }
            catch
            {
            }

            return counters.ToArray();
#pragma warning restore CA1416
        }

        private float[] SampleCpuCoreUsage()
        {
            var values = new[] { 0f, 0f, 0f, 0f };

            if (_cpuCoreCounters.Length == 0)
                return values;

            for (int i = 0; i < _cpuCoreCounters.Length && i < values.Length; i++)
            {
                try
                {
#pragma warning disable CA1416
                    values[i] = _cpuCoreCounters[i].NextValue();
#pragma warning restore CA1416
                }
                catch
                {
                    values[i] = 0f;
                }
            }

            return values;
        }

        public void Dispose()
        {
            Stop();
            foreach (var counter in _cpuCoreCounters)
            {
                try { counter.Dispose(); } catch { }
            }
            _cts.Dispose();
        }
    }
}
