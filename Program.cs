using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Xaviris.Core.Profiling;

namespace Xaviris.Core
{
    public class Program
    {
        private static readonly object CaptureOpenLock = new();

        private static class Config
        {
            public static string ModelPath { get; set; } = string.Empty;
            public static bool ShowDisplay { get; set; }
            public static int InferenceFps { get; set; }
            public static int CaptureSkip { get; set; }
            public static int DisplayWidth { get; set; }
            public static int DisplayHeight { get; set; }
            public static TimeSpan InferenceInterval { get; set; }
        }

        private static readonly MCvScalar GreenColor = new(0, 255, 0);
        private const int BoxThickness = 2;
        private const double FontScale = 0.5;
        private const int FontThickness = 1;

        private static List<string> GetStreamUrls(string[] args)
        { 
            if (args.Length > 0)
            {
                return new List<string>(args);
            }

            var urls = new List<string>();
            string? combinedUrl = Environment.GetEnvironmentVariable("XAVIRIS_STREAM_URL");
            if (!string.IsNullOrWhiteSpace(combinedUrl))
            {
                urls.AddRange(combinedUrl.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
            }

            if (urls.Count == 0)
            {
                urls.Add("udp://192.168.254.124:5000");
            }
            return urls;
        }

        private static async Task IngestStreamAsync(
            string streamUrl,
            FreshFrameBuffer frameBuffer,
            int captureSkip,
            CancellationToken cancellationToken)
        {
            const string captureOptions = "fflags;nobuffer|flags;low_delay|max_delay;0|fifo_size;1000000|overrun_nonfatal;1";

            while (!cancellationToken.IsCancellationRequested)
            {
                VideoCapture capture;
                try
                {
                    lock (CaptureOpenLock)
                    {
                        Environment.SetEnvironmentVariable("OPENCV_FFMPEG_CAPTURE_OPTIONS", captureOptions);
                        capture = new VideoCapture(streamUrl, VideoCapture.API.Ffmpeg);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Ingestion] OpenCV/FFmpeg error for {streamUrl}: {ex.Message}");
                    await Task.Delay(1000, cancellationToken).ConfigureAwait(false);
                    continue;
                }

                using (capture)
                {
                    if (!capture.IsOpened)
                    {
                        Console.WriteLine($"[Ingestion] Waiting for stream at {streamUrl}...");
                        await Task.Delay(1000, cancellationToken).ConfigureAwait(false);
                        continue;
                    }

                    Console.WriteLine($"[Ingestion] Connected: {streamUrl} ({capture.Width}x{capture.Height}).");
                    int frameNumber = 0;

                    while (!cancellationToken.IsCancellationRequested && capture.IsOpened)
                    {
                        if (!capture.Grab()) break;
                        frameNumber++;
                        if (frameNumber % captureSkip != 0) continue;

                        var frame = new Mat();
                        if (capture.Retrieve(frame) && !frame.IsEmpty)
                        {
                            frameBuffer.Push(frame);
                        }
                        else
                        {
                            frame.Dispose();
                            break;
                        }
                    }
                }
            }
        }

        public static async Task Main(string[] args)
        {
            var streamUrls = GetStreamUrls(args);

            // Initialize configuration (cached for entire runtime)
            string modelFileName = Environment.GetEnvironmentVariable("XAVIRIS_MODEL_PATH")
                ?? "yolov8n.onnx";
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelFileName);
            if (!File.Exists(modelPath))
            {
                modelPath = Path.Combine(Directory.GetCurrentDirectory(), modelFileName);
            }

            if (!File.Exists(modelPath))
            {
                Console.Error.WriteLine($"[Inference] Model not found: {modelPath}");
                Console.Error.WriteLine("Set XAVIRIS_MODEL_PATH to a YOLOv8 ONNX model.");
                return;
            }

            // Window preview is enabled by default here so you can see the stream.
            Config.ModelPath = modelPath;
            Config.ShowDisplay = true;
            Config.InferenceFps = int.TryParse(Environment.GetEnvironmentVariable("XAVIRIS_INFERENCE_FPS"), out int fps)
                ? Math.Clamp(fps, 1, 30) : 10;
            Config.CaptureSkip = int.TryParse(Environment.GetEnvironmentVariable("XAVIRIS_CAPTURE_SKIP"), out int skip)
                ? Math.Clamp(skip, 1, 5) : 2;
            Config.DisplayWidth = int.TryParse(Environment.GetEnvironmentVariable("XAVIRIS_DISPLAY_WIDTH"), out int width)
                ? Math.Clamp(width, 320, 1920) : 640;
            Config.InferenceInterval = TimeSpan.FromSeconds(1.0 / Config.InferenceFps);

            Console.WriteLine("=================================================");
            Console.WriteLine("XAVIRIS Max-Optimized Engine (UDP Only)");
            Console.WriteLine($"UDP Streams:   {string.Join(", ", streamUrls)}");
            Console.WriteLine($"Model Path:    {modelPath}");
            Console.WriteLine($"Display:       {(Config.ShowDisplay ? $"{Config.DisplayWidth}p" : "Headless")}");
            Console.WriteLine($"Inference FPS: {Config.InferenceFps}");
            Console.WriteLine("=================================================");

            foreach (string streamUrl in streamUrls)
            {
                if (!Uri.TryCreate(streamUrl, UriKind.Absolute, out Uri? streamUri) ||
                    !streamUri.Scheme.Equals("udp", StringComparison.OrdinalIgnoreCase) ||
                    streamUri.Port < 1)
                {
                    Console.Error.WriteLine($"[Ingestion] Invalid UDP stream URL: {streamUrl}");
                    return;
                }
            }

            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => { e.Cancel = true; cts.Cancel(); };

            bool telemetryEnabled = !"0".Equals(Environment.GetEnvironmentVariable("XAVIRIS_TELEMETRY_ENABLED"), StringComparison.OrdinalIgnoreCase);
            var telemetryProfiler = telemetryEnabled
                ? new BackgroundTelemetryProfiler(
                    csvPath: Environment.GetEnvironmentVariable("XAVIRIS_TELEMETRY_CSV") ??
                        Path.Combine(AppDomain.CurrentDomain.BaseDirectory, $"xaviris_telemetry_{DateTime.UtcNow:yyyyMMdd_HHmmss}.csv"),
                    samplingIntervalMs: 500)
                : null;

            if (telemetryProfiler != null)
            {
                for (int i = 0; i < streamUrls.Count; i++)
                {
                    telemetryProfiler.RegisterStream(i);
                }
            }

            using var frameBuffer = new FreshFrameBuffer();

            var ingestionTasks = new List<Task>(streamUrls.Count);
            foreach (string streamUrl in streamUrls)
            {
                ingestionTasks.Add(Task.Factory.StartNew(
                    () => IngestStreamAsync(streamUrl, frameBuffer, Config.CaptureSkip, cts.Token),
                    CancellationToken.None,
                    TaskCreationOptions.LongRunning,
                    TaskScheduler.Default).Unwrap());
            }

            var inferenceTask = Task.Factory.StartNew(() =>
            {
                using var engine = new StandardInferenceEngine(Config.ModelPath, 640, 640);
                using var renderCanvas = Config.ShowDisplay ? new Mat() : null;
                var lastStatus = Stopwatch.StartNew();
                var lastInference = Stopwatch.StartNew();
                int processedFrameNumber = 0;

                // Pre-cache display dimensions to avoid repeated calculations
                int displayWidth = Config.DisplayWidth;
                int displayHeight = 0;

                while (!cts.Token.IsCancellationRequested)
                {
                    Mat? frame = frameBuffer.WaitAndExtract(cts.Token);
                    if (frame == null) continue;

                    using (frame)
                    {
                        // Throttle inference to configured FPS
                        if (lastInference.Elapsed < Config.InferenceInterval)
                            continue;

                        lastInference.Restart();
                        var sw = Stopwatch.StartNew();
                        var detections = engine.Infer(frame, confidenceThreshold: 0.50f, iouThreshold: 0.45f);
                        sw.Stop();

                        if (telemetryProfiler != null)
                        {
                            var boxArray = new RectangleF[detections.Count];
                            var labelArray = new string[detections.Count];
                            for (int i = 0; i < detections.Count; i++)
                            {
                                boxArray[i] = detections[i].Box;
                                labelArray[i] = detections[i].Label;
                            }

                            var packet = telemetryProfiler.CreateFramePacket(
                                streamId: 0,
                                frameNumber: ++processedFrameNumber,
                                rawImage: "1".Equals(Environment.GetEnvironmentVariable("XAVIRIS_CAPTURE_TELEMETRY_IMAGES"), StringComparison.OrdinalIgnoreCase)
                                    ? frame.ToImage<Bgr, byte>().Bytes
                                    : Array.Empty<byte>(),
                                boundingBoxes: boxArray,
                                labels: labelArray,
                                fps: (float)(1.0 / Config.InferenceInterval.TotalSeconds),
                                latencyMs: sw.ElapsedMilliseconds,
                                packetDropRate: 0f);

                            telemetryProfiler.UpdateStreamTelemetry(0, packet.Fps, packet.ProcessingLatencyMs, packet.PacketDropRate);
                            _ = telemetryProfiler.PublishFrameAsync(packet, cts.Token);
                        }

                        foreach (var det in detections)
                        {
                            int left = Math.Clamp((int)det.Box.Left, 0, frame.Width - 1);
                            int top = Math.Clamp((int)det.Box.Top, 0, frame.Height - 1);
                            int right = Math.Clamp((int)det.Box.Right, left + 1, frame.Width);
                            int bottom = Math.Clamp((int)det.Box.Bottom, top + 1, frame.Height);
                            var rect = Rectangle.FromLTRB(left, top, right, bottom);

                            CvInvoke.Rectangle(frame, rect, GreenColor, BoxThickness);
                            CvInvoke.PutText(frame, $"{det.Label} {det.Confidence:P0}",
                                new Point(rect.X, Math.Max(15, rect.Y - 5)),
                                FontFace.HersheySimplex, FontScale, GreenColor, FontThickness);
                        }

                        if (Config.ShowDisplay)
                        {
                            if (displayHeight == 0)
                            {
                                displayHeight = Math.Max(1, (int)(frame.Height * (displayWidth / (double)frame.Width)));
                                Config.DisplayHeight = displayHeight;
                            }
                            CvInvoke.Resize(frame, renderCanvas, new Size(displayWidth, displayHeight));
                            CvInvoke.Imshow("XAVIRIS Max-Optimized Feed", renderCanvas);
                            CvInvoke.WaitKey(1);
                        }

                        if (lastStatus.Elapsed >= TimeSpan.FromSeconds(1))
                        {
                            Console.WriteLine($"[Inference] Latency: {sw.ElapsedMilliseconds}ms | Detections: {detections.Count}");
                            lastStatus.Restart();
                        }
                    }
                }
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

            try
            {
                ingestionTasks.Add(inferenceTask);
                await Task.WhenAll(ingestionTasks).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Clean shutdown
            }
            catch (Exception ex)
            {
                cts.Cancel();
                Console.Error.WriteLine($"[Runtime] Pipeline stopped: {ex.Message}");
            }
            finally
            {
                cts.Cancel();
            }

            if (Config.ShowDisplay)
            {
                CvInvoke.DestroyAllWindows();
            }
        }
    }
}