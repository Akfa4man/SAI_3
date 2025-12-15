using SAI_3.Data;
using System;
using System.Collections.Generic;

namespace SAI_3.Neural
{
    /// <summary>
    /// Простой многослойный перцептрон (MLP) для многоклассовой классификации:
    /// <para>
    /// <c>input → hidden (ReLU) → output (Softmax)</c>.
    /// </para>
    /// Реализация без фреймворков: все веса/смещения хранятся в массивах,
    /// прямой проход и backpropagation выполняются циклами.
    /// </summary>
    public sealed class Mlp
    {
        /// <summary>
        /// Размерность входного вектора (количество признаков).
        /// </summary>
        public int InputSize { get; }

        /// <summary>
        /// Количество нейронов скрытого слоя.
        /// </summary>
        public int HiddenSize { get; }

        /// <summary>
        /// Количество классов (размерность выхода).
        /// </summary>
        public int OutputSize { get; }

        /// <summary>
        /// Веса первого слоя (hidden × input): _w1[j][k] — вес от входа k к скрытому нейрону j.
        /// </summary>
        private readonly double[][] _w1;

        /// <summary>
        /// Смещения (bias) первого слоя (hidden): _b1[j] добавляется к сумме скрытого нейрона j.
        /// </summary>
        private readonly double[] _b1;

        /// <summary>
        /// Веса второго слоя (output × hidden): _w2[i][j] — вес от скрытого нейрона j к выходу i.
        /// </summary>
        private readonly double[][] _w2;

        /// <summary>
        /// Смещения (bias) второго слоя (output): _b2[i] добавляется к сумме выходного нейрона i.
        /// </summary>
        private readonly double[] _b2;

        /// <summary>
        /// Буфер предактиваций скрытого слоя: z1 = W1*x + b1.
        /// Хранится для вычисления производной ReLU при backpropagation.
        /// </summary>
        private readonly double[] _z1;

        /// <summary>
        /// Буфер активаций скрытого слоя: a1 = ReLU(z1).
        /// Используется и в прямом проходе, и при вычислении градиента W2.
        /// </summary>
        private readonly double[] _a1;

        /// <summary>
        /// Буфер логитов выходного слоя: z2 = W2*a1 + b2 (до Softmax).
        /// </summary>
        private readonly double[] _z2;

        /// <summary>
        /// Буфер вероятностей по классам: p = Softmax(z2).
        /// Используется для предсказания и вычисления cross-entropy.
        /// </summary>
        private readonly double[] _p;

        /// <summary>
        /// Буфер градиента по логитам выходного слоя: dz2 = dL/dz2.
        /// Для softmax + cross-entropy: dz2 = p; затем dz2[label] -= 1.
        /// </summary>
        private readonly double[] _dz2;

        /// <summary>
        /// Буфер градиента по активациям скрытого слоя: da1 = dL/da1.
        /// Получается умножением W2^T на dz2.
        /// </summary>
        private readonly double[] _da1;

        /// <summary>
        /// Буфер градиента по предактивациям скрытого слоя: dz1 = dL/dz1.
        /// Вычисляется как da1 * ReLU'(z1).
        /// </summary>
        private readonly double[] _dz1;

        /// <summary>
        /// Создаёт MLP с одним скрытым слоем и инициализирует веса.
        /// </summary>
        /// <param name="inputSize">Размерность входа (количество признаков).</param>
        /// <param name="hiddenSize">Число нейронов в скрытом слое.</param>
        /// <param name="outputSize">Число классов (размерность выхода). Должно быть &gt; 1.</param>
        /// <param name="seed">
        /// Seed для инициализации весов (детерминированность).
        /// Веса W1 и W2 инициализируются разными seed (seed и seed+1).
        /// </param>
        public Mlp(int inputSize, int hiddenSize, int outputSize, int seed = 42)
        {
            if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
            if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
            if (outputSize <= 1) throw new ArgumentOutOfRangeException(nameof(outputSize));

            InputSize = inputSize;
            HiddenSize = hiddenSize;
            OutputSize = outputSize;

            // Параметры модели.
            _w1 = AllocMatrix(hiddenSize, inputSize);
            _b1 = new double[hiddenSize];
            _w2 = AllocMatrix(outputSize, hiddenSize);
            _b2 = new double[outputSize];

            // Буферы прямого прохода.
            _z1 = new double[hiddenSize];
            _a1 = new double[hiddenSize];
            _z2 = new double[outputSize];
            _p = new double[outputSize];

            // Буферы backpropagation (переиспользуются между сэмплами).
            _dz2 = new double[outputSize];
            _da1 = new double[hiddenSize];
            _dz1 = new double[hiddenSize];

            // Инициализация He (под ReLU) — помогает избежать затухания/взрыва активаций на старте.
            InitWeightsHe(_w1, seed: seed);
            InitWeightsHe(_w2, seed: seed + 1);
        }

        /// <summary>
        /// Возвращает вероятности по классам для входного вектора.
        /// <para>
        /// Делает прямой проход и возвращает копию буфера softmax-вероятностей.
        /// </para>
        /// </summary>
        /// <param name="x">Входной вектор длиной <see cref="InputSize"/>.</param>
        /// <returns>Новый массив длиной <see cref="OutputSize"/> (копия softmax-вероятностей).</returns>
        public double[] PredictProba(double[] x)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Length != InputSize) throw new ArgumentException("Bad input size", nameof(x));

            Forward(x);

            // Возвращаем копию, чтобы вызывающий код не мог случайно изменить внутренний буфер _p.
            var copy = new double[OutputSize];
            Array.Copy(_p, copy, OutputSize);
            return copy;
        }

        /// <summary>
        /// Возвращает предсказанный класс (индекс максимальной вероятности).
        /// </summary>
        /// <param name="x">Входной вектор длиной <see cref="InputSize"/>.</param>
        /// <returns>Индекс класса (0..OutputSize-1).</returns>
        public int Predict(double[] x)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (x.Length != InputSize) throw new ArgumentException("Bad input size", nameof(x));

            Forward(x);
            return MathHelpers.ArgMax(_p);
        }

        /// <summary>
        /// Обучает модель одну эпоху на датасете методом mini-batch SGD.
        /// <para>
        /// Данные перемешиваются, затем последовательно формируются батчи,
        /// для каждого батча вызывается <see cref="TrainBatch(double[][], int[], double)"/>.
        /// </para>
        /// </summary>
        /// <param name="data">Датасет (список примеров).</param>
        /// <param name="batchSize">Размер батча (&gt; 0).</param>
        /// <param name="learningRate">Шаг обучения (&gt; 0).</param>
        /// <param name="rng">Генератор случайных чисел для перемешивания.</param>
        /// <returns>Средний loss (cross-entropy) по батчам за эпоху.</returns>
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
        /// Обучает модель на одном батче.
        /// <para>
        /// Функция потерь: cross-entropy, выход: softmax.
        /// Градиенты аккумулируются по всем примерам батча и затем усредняются (деление на N)
        /// перед шагом обновления параметров.
        /// </para>
        /// </summary>
        /// <param name="batchX">Массив входов, каждый элемент длиной <see cref="InputSize"/>.</param>
        /// <param name="batchY">Массив меток классов (0..OutputSize-1).</param>
        /// <param name="learningRate">Шаг обучения (&gt; 0).</param>
        /// <returns>Средний loss (cross-entropy) по батчу.</returns>
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

            // Проходим по всем примерам батча: forward → loss → backward → accumulate grads.
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

                // Градиент по логитам: dz2 = p; dz2[label] -= 1.
                for (int i = 0; i < OutputSize; i++)
                    _dz2[i] = _p[i];
                _dz2[label] -= 1.0;

                // dW2 и db2: dW2[i][j] += dz2[i] * a1[j], db2[i] += dz2[i]
                for (int i = 0; i < OutputSize; i++)
                {
                    db2[i] += _dz2[i];
                    var dRow = dW2[i];
                    for (int j = 0; j < HiddenSize; j++)
                        dRow[j] += _dz2[i] * _a1[j];
                }

                // da1 = W2^T * dz2
                for (int j = 0; j < HiddenSize; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < OutputSize; i++)
                        sum += _w2[i][j] * _dz2[i];
                    _da1[j] = sum;
                }

                // dz1 = da1 * ReLU'(z1)
                for (int j = 0; j < HiddenSize; j++)
                    _dz1[j] = _da1[j] * MathHelpers.ReLUDerivative(_z1[j]);

                // dW1 и db1: dW1[j][k] += dz1[j] * x[k], db1[j] += dz1[j]
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
        /// Оценивает точность (accuracy) на датасете:
        /// доля примеров, для которых <see cref="Predict(double[])"/> совпал с меткой.
        /// </summary>
        /// <param name="data">Датасет для оценки.</param>
        /// <returns>Значение accuracy в диапазоне [0; 1].</returns>
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

        /// <summary>
        /// Прямой проход сети для одного примера:
        /// <para>
        /// z1 = W1*x + b1; a1 = ReLU(z1); z2 = W2*a1 + b2; p = Softmax(z2).
        /// </para>
        /// Результаты складываются во внутренние буферы (_z1/_a1/_z2/_p),
        /// чтобы затем использовать их в backpropagation без дополнительных выделений памяти.
        /// </summary>
        /// <param name="x">Входной вектор длиной <see cref="InputSize"/>.</param>
        private void Forward(double[] x)
        {
            // Скрытый слой.
            for (int j = 0; j < HiddenSize; j++)
            {
                double sum = _b1[j];
                var row = _w1[j];
                for (int k = 0; k < InputSize; k++)
                    sum += row[k] * x[k];

                _z1[j] = sum;
                _a1[j] = MathHelpers.ReLU(sum);
            }

            // Выходной слой (логиты).
            for (int i = 0; i < OutputSize; i++)
            {
                double sum = _b2[i];
                var row = _w2[i];
                for (int j = 0; j < HiddenSize; j++)
                    sum += row[j] * _a1[j];

                _z2[i] = sum;
            }

            // Преобразуем логиты в вероятности в буфер _p.
            SoftmaxToBuffer(_z2, _p);
        }

        /// <summary>
        /// Вычисляет softmax(logits) и записывает результат в заранее выделенный буфер.
        /// <para>
        /// Используется внутри <see cref="Forward(double[])"/> для снижения аллокаций.
        /// Реализация численно стабильная: вычитает максимум из логитов.
        /// </para>
        /// </summary>
        /// <param name="logits">Входные логиты (длина = число классов).</param>
        /// <param name="probs">Буфер для вероятностей (та же длина, что и <paramref name="logits"/>).</param>
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

        /// <summary>
        /// Выделяет "рваную" матрицу double[rows][cols].
        /// Используется для хранения весов и их градиентов.
        /// </summary>
        private static double[][] AllocMatrix(int rows, int cols)
        {
            var m = new double[rows][];
            for (int r = 0; r < rows; r++)
                m[r] = new double[cols];
            return m;
        }

        /// <summary>
        /// Инициализация весов по He (Kaiming) для слоёв с ReLU.
        /// <para>
        /// Каждый вес берётся как N(0, 1) * sqrt(2/fanIn),
        /// где fanIn — число входов в нейрон (число столбцов матрицы).
        /// </para>
        /// </summary>
        /// <param name="w">Матрица весов (rows × cols), заполняется на месте.</param>
        /// <param name="seed">Seed для детерминированной инициализации.</param>
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

        /// <summary>
        /// Возвращает глубокую копию текущих параметров сети (веса и bias).
        /// <para>
        /// Полезно для сохранения модели, вывода в UI, сериализации и т.п.
        /// </para>
        /// </summary>
        /// <returns>Копии (W1, B1, W2, B2), изменение которых не влияет на модель.</returns>
        public (double[][] W1, double[] B1, double[][] W2, double[] B2) GetParametersCopy()
        {
            return (CopyMatrix(_w1), (double[])_b1.Clone(), CopyMatrix(_w2), (double[])_b2.Clone());
        }

        /// <summary>
        /// Полностью заменяет параметры сети на переданные (копированием в внутренние массивы).
        /// <para>
        /// Используется при загрузке модели/обновлении параметров извне.
        /// </para>
        /// </summary>
        /// <param name="w1">Матрица весов первого слоя (hidden × input).</param>
        /// <param name="b1">Bias первого слоя (hidden).</param>
        /// <param name="w2">Матрица весов второго слоя (output × hidden).</param>
        /// <param name="b2">Bias второго слоя (output).</param>
        /// <exception cref="ArgumentNullException">Если какой-либо параметр равен null.</exception>
        /// <exception cref="ArgumentException">Если размерности параметров не совпадают с конфигурацией сети.</exception>
        public void SetParameters(double[][] w1, double[] b1, double[][] w2, double[] b2)
        {
            if (w1 == null || b1 == null || w2 == null || b2 == null)
                throw new ArgumentNullException("Parameters are null");

            if (w1.Length != HiddenSize || w1[0].Length != InputSize) throw new ArgumentException("Bad W1 shape");
            if (b1.Length != HiddenSize) throw new ArgumentException("Bad b1 length");
            if (w2.Length != OutputSize || w2[0].Length != HiddenSize) throw new ArgumentException("Bad W2 shape");
            if (b2.Length != OutputSize) throw new ArgumentException("Bad b2 length");

            // Копируем значения в уже выделенные массивы, чтобы не менять ссылки на поля.
            for (int i = 0; i < HiddenSize; i++)
                Array.Copy(w1[i], _w1[i], InputSize);
            Array.Copy(b1, _b1, HiddenSize);

            for (int i = 0; i < OutputSize; i++)
                Array.Copy(w2[i], _w2[i], HiddenSize);
            Array.Copy(b2, _b2, OutputSize);
        }

        /// <summary>
        /// Создаёт глубокую копию матрицы (jagged array).
        /// </summary>
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