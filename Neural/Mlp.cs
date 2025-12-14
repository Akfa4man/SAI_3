using SAI_3.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SAI_3.Neural
{
    /// <summary>
    /// Простой MLP: input -> hidden(ReLU) -> output(Softmax).
    /// Без слоёв/оптимизаторов/фреймворков — только базовые массивы и циклы.
    /// </summary>
    public sealed class Mlp
    {
        public int InputSize { get; }
        public int HiddenSize { get; }
        public int OutputSize { get; }

        private readonly double[][] _w1;
        private readonly double[] _b1;
        private readonly double[][] _w2;
        private readonly double[] _b2;

        private readonly double[] _z1;
        private readonly double[] _a1;
        private readonly double[] _z2;
        private readonly double[] _p;

        private readonly double[] _dz2;
        private readonly double[] _da1;
        private readonly double[] _dz1;

        public Mlp(int inputSize, int hiddenSize, int outputSize, int seed = 42)
        {
            if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
            if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
            if (outputSize <= 1) throw new ArgumentOutOfRangeException(nameof(outputSize));

            InputSize = inputSize;
            HiddenSize = hiddenSize;
            OutputSize = outputSize;

            _w1 = AllocMatrix(hiddenSize, inputSize);
            _b1 = new double[hiddenSize];
            _w2 = AllocMatrix(outputSize, hiddenSize);
            _b2 = new double[outputSize];

            _z1 = new double[hiddenSize];
            _a1 = new double[hiddenSize];
            _z2 = new double[outputSize];
            _p = new double[outputSize];

            _dz2 = new double[outputSize];
            _da1 = new double[hiddenSize];
            _dz1 = new double[hiddenSize];

            InitWeightsHe(_w1, seed: seed);
            InitWeightsHe(_w2, seed: seed + 1);
        }

        /// <summary>
        /// Возвращает вероятности по классам (softmax).
        /// </summary>
        public double[] PredictProba(double[] x)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Length != InputSize) throw new ArgumentException("Bad input size", nameof(x));

            Forward(x);

            var copy = new double[OutputSize];
            Array.Copy(_p, copy, OutputSize);
            return copy;
        }

        /// <summary>
        /// Предсказанный класс (argmax softmax).
        /// </summary>
        public int Predict(double[] x)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Length != InputSize) throw new ArgumentException("Bad input size", nameof(x));

            Forward(x);
            return MathHelpers.ArgMax(_p);
        }

        /// <summary>
        /// Одна эпоха обучения по данным (SGD mini-batch).
        /// Возвращает средний loss.
        /// </summary>
        public double TrainEpoch(List<Sample> data, int batchSize, double learningRate, Random rng)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (data.Count == 0) throw new ArgumentException("Dataset is empty", nameof(data));
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
            if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));
            if (rng == null) throw new ArgumentNullException(nameof(rng));

            MathHelpers.Shuffle(data, rng);

            double totalLoss = 0.0;
            int batches = 0;

            for (int i = 0; i < data.Count; i += batchSize)
            {
                int n = Math.Min(batchSize, data.Count - i);

                var batchX = new double[n][];
                var batchY = new int[n];
                for (int j = 0; j < n; j++)
                {
                    batchX[j] = data[i + j].X;
                    batchY[j] = data[i + j].Label;
                }

                totalLoss += TrainBatch(batchX, batchY, learningRate);
                batches++;
            }

            return totalLoss / Math.Max(1, batches);
        }

        /// <summary>
        /// Тренировка на одном batch, возвращает average loss по batch.
        /// Cross-entropy + softmax.
        /// </summary>
        public double TrainBatch(double[][] batchX, int[] batchY, double learningRate)
        {
            if (batchX == null) throw new ArgumentNullException(nameof(batchX));
            if (batchY == null) throw new ArgumentNullException(nameof(batchY));
            if (batchX.Length != batchY.Length) throw new ArgumentException("X/Y size mismatch");
            if (batchX.Length == 0) throw new ArgumentException("Empty batch");
            if (learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));

            int n = batchX.Length;

            var dW1 = AllocMatrix(HiddenSize, InputSize);
            var db1 = new double[HiddenSize];
            var dW2 = AllocMatrix(OutputSize, HiddenSize);
            var db2 = new double[OutputSize];

            double lossSum = 0.0;

            for (int s = 0; s < n; s++)
            {
                double[] x = batchX[s];
                int label = batchY[s];

                if (x.Length != InputSize)
                    throw new ArgumentException("Bad input size in batch");

                if (label < 0 || label >= OutputSize)
                    throw new ArgumentOutOfRangeException(nameof(batchY), "Label out of range");

                Forward(x);

                lossSum += -Math.Log(Math.Max(1e-12, _p[label]));

                for (int i = 0; i < OutputSize; i++)
                    _dz2[i] = _p[i];
                _dz2[label] -= 1.0;

                for (int i = 0; i < OutputSize; i++)
                {
                    db2[i] += _dz2[i];
                    var dRow = dW2[i];
                    for (int j = 0; j < HiddenSize; j++)
                        dRow[j] += _dz2[i] * _a1[j];
                }

                for (int j = 0; j < HiddenSize; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < OutputSize; i++)
                        sum += _w2[i][j] * _dz2[i];
                    _da1[j] = sum;
                }

                for (int j = 0; j < HiddenSize; j++)
                    _dz1[j] = _da1[j] * MathHelpers.ReLUDerivative(_z1[j]);

                for (int j = 0; j < HiddenSize; j++)
                {
                    db1[j] += _dz1[j];
                    var dRow = dW1[j];
                    for (int k = 0; k < InputSize; k++)
                        dRow[k] += _dz1[j] * x[k];
                }
            }

            double invN = 1.0 / n;

            for (int j = 0; j < HiddenSize; j++)
            {
                _b1[j] -= learningRate * db1[j] * invN;
                for (int k = 0; k < InputSize; k++)
                    _w1[j][k] -= learningRate * dW1[j][k] * invN;
            }

            for (int i = 0; i < OutputSize; i++)
            {
                _b2[i] -= learningRate * db2[i] * invN;
                for (int j = 0; j < HiddenSize; j++)
                    _w2[i][j] -= learningRate * dW2[i][j] * invN;
            }

            return lossSum * invN;
        }

        /// <summary>
        /// Точность (accuracy) на датасете.
        /// </summary>
        public double EvaluateAccuracy(List<Sample> data)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (data.Count == 0) return 0.0;

            int ok = 0;
            foreach (var s in data)
            {
                if (Predict(s.X) == s.Label)
                    ok++;
            }

            return (double)ok / data.Count;
        }

        private void Forward(double[] x)
        {
            for (int j = 0; j < HiddenSize; j++)
            {
                double sum = _b1[j];
                var row = _w1[j];
                for (int k = 0; k < InputSize; k++)
                    sum += row[k] * x[k];

                _z1[j] = sum;
                _a1[j] = MathHelpers.ReLU(sum);
            }

            for (int i = 0; i < OutputSize; i++)
            {
                double sum = _b2[i];
                var row = _w2[i];
                for (int j = 0; j < HiddenSize; j++)
                    sum += row[j] * _a1[j];

                _z2[i] = sum;
            }

            SoftmaxToBuffer(_z2, _p);
        }

        private static void SoftmaxToBuffer(double[] logits, double[] probs)
        {
            double max = logits[0];
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > max) max = logits[i];

            double sum = 0.0;
            for (int i = 0; i < logits.Length; i++)
            {
                double e = Math.Exp(logits[i] - max);
                probs[i] = e;
                sum += e;
            }

            double inv = 1.0 / Math.Max(1e-12, sum);
            for (int i = 0; i < probs.Length; i++)
                probs[i] *= inv;
        }

        private static double[][] AllocMatrix(int rows, int cols)
        {
            var m = new double[rows][];
            for (int r = 0; r < rows; r++)
                m[r] = new double[cols];
            return m;
        }

        private static void InitWeightsHe(double[][] w, int seed)
        {
            var rng = new Random(seed);
            int fanIn = w[0].Length;
            double std = Math.Sqrt(2.0 / fanIn);

            for (int i = 0; i < w.Length; i++)
            {
                for (int j = 0; j < w[i].Length; j++)
                    w[i][j] = MathHelpers.NextGaussian(rng) * std;
            }
        }

        public (double[][] W1, double[] B1, double[][] W2, double[] B2) GetParametersCopy()
        {
            return (CopyMatrix(_w1), (double[])_b1.Clone(), CopyMatrix(_w2), (double[])_b2.Clone());
        }

        public void SetParameters(double[][] w1, double[] b1, double[][] w2, double[] b2)
        {
            if (w1 == null || b1 == null || w2 == null || b2 == null)
                throw new ArgumentNullException("Parameters are null");

            if (w1.Length != HiddenSize || w1[0].Length != InputSize) throw new ArgumentException("Bad W1 shape");
            if (b1.Length != HiddenSize) throw new ArgumentException("Bad b1 length");
            if (w2.Length != OutputSize || w2[0].Length != HiddenSize) throw new ArgumentException("Bad W2 shape");
            if (b2.Length != OutputSize) throw new ArgumentException("Bad b2 length");

            for (int i = 0; i < HiddenSize; i++)
                Array.Copy(w1[i], _w1[i], InputSize);
            Array.Copy(b1, _b1, HiddenSize);

            for (int i = 0; i < OutputSize; i++)
                Array.Copy(w2[i], _w2[i], HiddenSize);
            Array.Copy(b2, _b2, OutputSize);
        }

        private static double[][] CopyMatrix(double[][] src)
        {
            var dst = new double[src.Length][];
            for (int i = 0; i < src.Length; i++)
            {
                dst[i] = new double[src[i].Length];
                Array.Copy(src[i], dst[i], src[i].Length);
            }
            return dst;
        }
    }
}
