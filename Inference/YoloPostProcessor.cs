using System;
using System.Buffers;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace Xaviris.Core
{


    public readonly struct Detection
    {

        public readonly RectangleF Box;

        public readonly float Confidence;

        public readonly int ClassId;


        public readonly string Label;

        public Detection(float x, float y, float width, float height, float confidence, int classId = 0, string label = "Unknown")
        {
            Box = new RectangleF(x, y, width, height);
            Confidence = confidence;
            ClassId = classId;
            Label = label;
        }
    }


    public static class YoloPostProcessor
    {
        private const string PersonLabel = "Person";

        public static string GetLabel(int classId)
        {
            return classId == 0 ? PersonLabel : "Unknown";
        }

        public static int GetClassCount()
        {
            return 1;
        }

        public static List<Detection> ProcessOutput(
            ReadOnlySpan<float> output,
            float confidenceThreshold = 0.50f,
            float iouThreshold = 0.45f,
            int targetClassId = 0)
        {
            const int numAnchors = 8400;
            int classCount = (output.Length / numAnchors) - 4;
            if (classCount <= targetClassId)
            {
                return new List<Detection>();
            }

            var candidates = new List<Detection>(64);

            for (int a = 0; a < numAnchors; a++)
            {
                float classScore = output[(4 + targetClassId) * numAnchors + a];
                if (classScore < confidenceThreshold)
                {
                    continue;
                }

                float cx = output[0 * numAnchors + a];
                float cy = output[1 * numAnchors + a];
                float w = output[2 * numAnchors + a];
                float h = output[3 * numAnchors + a];

                candidates.Add(new Detection(
                    cx - (w * 0.5f),
                    cy - (h * 0.5f),
                    w,
                    h,
                    classScore,
                    targetClassId,
                    GetLabel(targetClassId)));
            }

            return ApplyZeroAllocNMS(candidates, iouThreshold);
        }

        private static List<Detection> ApplyZeroAllocNMS(List<Detection> boxes, float iouThreshold)
        {
            int count = boxes.Count;
            if (count == 0) return boxes;

            boxes.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

            bool[] active = ArrayPool<bool>.Shared.Rent(count);
            Array.Fill(active, true, 0, count);

            var result = new List<Detection>(count);

            try
            {
                for (int i = 0; i < count; i++)
                {
                    if (!active[i]) continue;

                    var current = boxes[i];
                    result.Add(current);

                    for (int j = i + 1; j < count; j++)
                    {
                        if (!active[j]) continue;

                        if (CalculateIoU(current.Box, boxes[j].Box) > iouThreshold)
                        {
                            active[j] = false;
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<bool>.Shared.Return(active);
            }

            return result;
        }

        /// <summary>
        private static float CalculateIoU(in RectangleF a, in RectangleF b)
        {
            float x1 = Math.Max(a.X, b.X);
            float y1 = Math.Max(a.Y, b.Y);
            float x2 = Math.Min(a.Right, b.Right);
            float y2 = Math.Min(a.Bottom, b.Bottom);

            float intersection = Math.Max(0f, x2 - x1) * Math.Max(0f, y2 - y1);
            float union = (a.Width * a.Height) + (b.Width * b.Height) - intersection;

            return union <= 0f ? 0f : intersection / union;
        }
    }
}