using SAI_3.Neural;
using System;
using System.IO;
using System.Text.Json;

namespace SAI_3
{
    /// <summary>
    /// Сериализация/десериализация параметров нейросети <see cref="Mlp"/> в JSON-файл.
    /// <para>
    /// Сохраняются размеры сети (Input/Hidden/Output) и параметры слоёв (W1/B1/W2/B2).
    /// </para>
    /// <para>
    /// Предназначено для простого сценария: быстро сохранить обученную модель и затем
    /// восстановить её без повторного обучения.
    /// </para>
    /// </summary>
    public static class ModelIO
    {
        /// <summary>
        /// Внутренний DTO-формат модели для JSON.
        /// <para>
        /// Используется только для хранения данных и сериализации, без логики.
        /// </para>
        /// </summary>
        private sealed class ModelDto
        {
            /// <summary>
            /// Размер входа (количество признаков).
            /// </summary>
            public int Input { get; set; }

            /// <summary>
            /// Размер скрытого слоя (число нейронов).
            /// </summary>
            public int Hidden { get; set; }

            /// <summary>
            /// Размер выхода (число классов).
            /// </summary>
            public int Output { get; set; }

            /// <summary>
            /// Веса первого слоя (hidden × input).
            /// </summary>
            public double[][] W1 { get; set; } = Array.Empty<double[]>();

            /// <summary>
            /// Bias первого слоя (hidden).
            /// </summary>
            public double[] B1 { get; set; } = Array.Empty<double>();

            /// <summary>
            /// Веса второго слоя (output × hidden).
            /// </summary>
            public double[][] W2 { get; set; } = Array.Empty<double[]>();

            /// <summary>
            /// Bias второго слоя (output).
            /// </summary>
            public double[] B2 { get; set; } = Array.Empty<double>();
        }

        /// <summary>
        /// Сохраняет параметры сети в JSON-файл по указанному пути.
        /// <para>
        /// Использует <see cref="Mlp.GetParametersCopy"/> — т.е. сохраняются копии параметров,
        /// а не ссылки на внутренние массивы сети.
        /// </para>
        /// <para>
        /// Если директория в <paramref name="path"/> отсутствует — она будет создана.
        /// </para>
        /// </summary>
        /// <param name="net">Экземпляр сети, параметры которой нужно сохранить.</param>
        /// <param name="path">Путь к файлу модели (например, <c>Models/model.json</c>).</param>
        public static void Save(Mlp net, string path)
        {
            if (net == null) throw new ArgumentNullException(nameof(net));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Bad path", nameof(path));

            // Берём копии параметров, чтобы сериализация не зависела от текущего состояния внутренних буферов.
            var (w1, b1, w2, b2) = net.GetParametersCopy();

            var dto = new ModelDto
            {
                Input = net.InputSize,
                Hidden = net.HiddenSize,
                Output = net.OutputSize,
                W1 = w1,
                B1 = b1,
                W2 = w2,
                B2 = b2
            };

            var opt = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(dto, opt);

            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllText(path, json);
        }

        /// <summary>
        /// Загружает модель из JSON-файла и возвращает новый экземпляр <see cref="Mlp"/>
        /// с восстановленными параметрами.
        /// </summary>
        /// <param name="path">Путь к файлу модели (JSON).</param>
        /// <returns>Новый экземпляр <see cref="Mlp"/> с загруженными параметрами.</returns>
        public static Mlp Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Model file not found", path);

            var json = File.ReadAllText(path);

            // Если структура JSON не соответствует ожидаемой, десериализация вернёт null.
            var dto = JsonSerializer.Deserialize<ModelDto>(json)
                ?? throw new InvalidOperationException("Bad model json");

            var net = new Mlp(dto.Input, dto.Hidden, dto.Output, seed: 1);
            net.SetParameters(dto.W1, dto.B1, dto.W2, dto.B2);
            return net;
        }
    }
}