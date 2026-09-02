using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime;

namespace Xaviris.Core
{
    public sealed class StandardInferenceEngine : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _modelWidth;
        private readonly int _modelHeight;
        private readonly float[] _rawInputBuffer;
        private readonly Mat _resized;
        private readonly Mat _letterboxedMat;
        private readonly long[] _inputDims;
        private readonly OrtValue _inputOrtValue;
        private readonly Dictionary<string, OrtValue> _inputs;
        private readonly string _inputName;
        private readonly IReadOnlyList<string> _outputNames;
        private readonly ParallelOptions _parallelOptions;
        private readonly RunOptions _runOptions;
        private readonly int _targetClassId;

        public StandardInferenceEngine(string modelPath, int modelWidth = 640, int modelHeight = 640)
        {
            _modelWidth = modelWidth;
            _modelHeight = modelHeight;
            _targetClassId = 0;

            int cpuThreads = Math.Clamp(
                int.TryParse(Environment.GetEnvironmentVariable("XAVIRIS_CPU_THREADS"), out int configuredThreads)
                    ? configuredThreads
                    : 4,
                1,
                Math.Max(1, Environment.ProcessorCount));

            var sessionOptions = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                IntraOpNumThreads = cpuThreads,
                InterOpNumThreads = 1
            };

            try
            {
                sessionOptions.AppendExecutionProvider_DML(0);
            }
            catch
            {
            }

            _session = new InferenceSession(modelPath, sessionOptions);
            _rawInputBuffer = GC.AllocateArray<float>(1 * 3 * _modelWidth * _modelHeight, pinned: true);
            _resized = new Mat();
            _letterboxedMat = new Mat(_modelHeight, _modelWidth, DepthType.Cv8U, 3);
            _inputDims = new long[] { 1, 3, _modelHeight, _modelWidth };
            _inputName = _session.InputNames[0];
            _outputNames = _session.OutputNames;
            _inputOrtValue = OrtValue.CreateTensorValueFromMemory(_rawInputBuffer, _inputDims);
            _inputs = new Dictionary<string, OrtValue>(1);
            _inputs.Add(_inputName, _inputOrtValue);
            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = cpuThreads };
            _runOptions = new RunOptions();
        }

        public unsafe List<Detection> Infer(Mat frame, float confidenceThreshold = 0.50f, float iouThreshold = 0.45f)
        {
            float scale = Math.Min((float)_modelWidth / frame.Width, (float)_modelHeight / frame.Height);
            int newUnpadW = (int)Math.Round(frame.Width * scale);
            int newUnpadH = (int)Math.Round(frame.Height * scale);

            int padX = (_modelWidth - newUnpadW) / 2;
            int padY = (_modelHeight - newUnpadH) / 2;

            CvInvoke.Resize(frame, _resized, new Size(newUnpadW, newUnpadH), 0, 0, Inter.Linear);

            int top = padY;
            int bottom = _modelHeight - newUnpadH - padY;
            int left = padX;
            int right = _modelWidth - newUnpadW - padX;

            CvInvoke.CopyMakeBorder(_resized, _letterboxedMat, top, bottom, left, right, BorderType.Constant, new MCvScalar(114, 114, 114));

            byte* srcPtr = (byte*)_letterboxedMat.DataPointer;
            int step = _letterboxedMat.Step;
            int width = _modelWidth;
            int height = _modelHeight;
            int planeSize = width * height;
            int strideG = planeSize;
            int strideB = planeSize * 2;
            const float normScale = 1.0f / 255.0f;

            fixed (float* dstPtr = _rawInputBuffer)
            {
                nint dstAddress = (nint)dstPtr;
                nint srcAddress = (nint)srcPtr;

                Parallel.For(0, height, _parallelOptions, y =>
                {
                    byte* localSrcPtr = (byte*)srcAddress;
                    float* localDstPtr = (float*)dstAddress;

                    int rowByteOffset = y * step;
                    int rowPixelOffset = y * width;
                    byte* rowSrc = localSrcPtr + rowByteOffset;

                    float* dstR = localDstPtr + rowPixelOffset;
                    float* dstG = localDstPtr + strideG + rowPixelOffset;
                    float* dstB = localDstPtr + strideB + rowPixelOffset;

                    for (int x = 0; x < width; x++)
                    {
                        int pxIdx = x * 3;
                        dstR[x] = rowSrc[pxIdx + 2] * normScale;
                        dstG[x] = rowSrc[pxIdx + 1] * normScale;
                        dstB[x] = rowSrc[pxIdx]     * normScale;
                    }
                });
            }

            using var outputs = _session.Run(_runOptions, _inputs, _outputNames);

            ReadOnlySpan<float> rawOutput = outputs[0].GetTensorDataAsSpan<float>();
            var rawDetections = YoloPostProcessor.ProcessOutput(
                rawOutput, confidenceThreshold, iouThreshold, _targetClassId);

            var scaledDetections = new List<Detection>(rawDetections.Count);
            foreach (var d in rawDetections)
            {
                float unpaddedX = (d.Box.X - padX) / scale;
                float unpaddedY = (d.Box.Y - padY) / scale;
                float unpaddedW = d.Box.Width / scale;
                float unpaddedH = d.Box.Height / scale;

                scaledDetections.Add(new Detection(unpaddedX, unpaddedY, unpaddedW, unpaddedH, d.Confidence));
            }

            return scaledDetections;
        }

        public void Dispose()
        {
            _inputOrtValue.Dispose();
            _resized.Dispose();
            _letterboxedMat.Dispose();
            _runOptions?.Dispose();
            _session?.Dispose();
        }
    }
}