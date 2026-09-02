using System;
using System.Threading;
using Emgu.CV;

namespace Xaviris.Core
{
    public sealed class FreshFrameBuffer : IDisposable
    {
        private Mat? _activeFrame;
        private Mat? _stagingFrame;
        private readonly SemaphoreSlim _frameAvailable;
        private readonly object _lock = new();
        
        public long FramesReceived { get; private set; }
        public long FramesDropped { get; private set; }
        public long FramesExtracted { get; private set; }
        private DateTime _lastPushTime;
        private DateTime _lastExtractTime;

        public FreshFrameBuffer()
        {
            // Allow multiple pending frames without overflowing the signal gate.
            // The buffer itself drops stale frames, but the semaphore must not be artificially capped at 1.
            _frameAvailable = new SemaphoreSlim(0, int.MaxValue);
            _lastPushTime = DateTime.UtcNow;
            _lastExtractTime = DateTime.UtcNow;
        }

        public void Push(Mat newFrame)
        {
            lock (_lock)
            {
                if (_stagingFrame == null)
                {
                    _stagingFrame = newFrame;
                }
                else
                {
                    var oldActive = _activeFrame;
                    _activeFrame = _stagingFrame;
                    _stagingFrame = newFrame;
                    oldActive?.Dispose();
                    FramesDropped++;
                }

                FramesReceived++;
                _lastPushTime = DateTime.UtcNow;
            }

            _frameAvailable.Release();
        }

        public Mat? WaitAndExtract(CancellationToken cancellationToken)
        {
            try
            {
                if (!_frameAvailable.Wait(Timeout.Infinite, cancellationToken))
                    return null;

                lock (_lock)
                {
                    // Promote staging to active if needed
                    if (_stagingFrame != null && _activeFrame == null)
                    {
                        _activeFrame = _stagingFrame;
                        _stagingFrame = null;
                    }

                    Mat? frame = _activeFrame;
                    _activeFrame = null;

                    if (frame != null)
                    {
                        FramesExtracted++;
                        _lastExtractTime = DateTime.UtcNow;
                    }

                    return frame;
                }
            }
            catch (OperationCanceledException)
            {
                return null;
            }
        }

        public BufferMetrics GetMetrics()
        {
            lock (_lock)
            {
                long dropRate = FramesReceived > 0 ? (FramesDropped * 100) / FramesReceived : 0;
                bool hasFrame = _activeFrame != null || _stagingFrame != null;
                double avgLatency = _lastExtractTime > _lastPushTime 
                    ? 0 
                    : (_lastPushTime - _lastExtractTime).TotalMilliseconds;

                return new BufferMetrics(
                    FramesReceived,
                    FramesExtracted,
                    FramesDropped,
                    dropRate,
                    hasFrame,
                    avgLatency);
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                _activeFrame?.Dispose();
                _activeFrame = null;
                _stagingFrame?.Dispose();
                _stagingFrame = null;
            }
            _frameAvailable?.Dispose();
        }
    }


    public readonly struct BufferMetrics
    {
        public readonly long TotalReceived;
        public readonly long TotalExtracted;
        public readonly long TotalDropped;
        public readonly long DropRatePercent;
        public readonly bool HasPendingFrame;
        public readonly double LatencyMs;

        public BufferMetrics(long received, long extracted, long dropped, long dropRate, bool hasPending, double latency)
        {
            TotalReceived = received;
            TotalExtracted = extracted;
            TotalDropped = dropped;
            DropRatePercent = dropRate;
            HasPendingFrame = hasPending;
            LatencyMs = latency;
        }

        public override string ToString()
        {
            return $"[Buffer] Received: {TotalReceived} | Extracted: {TotalExtracted} | Dropped: {TotalDropped} ({DropRatePercent}%) | Latency: {LatencyMs:F2}ms | Pending: {HasPendingFrame}";
        }
    }
}