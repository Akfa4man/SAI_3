using SAI_3.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace SAI_3
{
    /// <summary>
    /// Утилита для хранения пользовательских примеров (<see cref="Sample"/>) в JSON-файле.
    /// <para>
    /// Формат: массив объектов вида { label: int, x: double[] }.
    /// Используется для накопления/подгрузки данных для дообучения или тестирования.
    /// </para>
    /// </summary>
    public static class SamplesIO
    {
        /// <summary>
        /// Внутренний DTO-формат одного примера для JSON.
        /// <para>
        /// Нужен только для сериализации/десериализации, без логики.
        /// </para>
        /// </summary>
        private sealed class SampleDto
        {
            /// <summary>
            /// Метка класса.
            /// </summary>
            public int Label { get; set; }

            /// <summary>
            /// Входной вектор признаков.
            /// </summary>
            public double[] X { get; set; } = Array.Empty<double>();
        }

        /// <summary>
        /// Добавляет один пример в файл с примерами.
        /// <para>
        /// Реализовано в простом виде: загружает весь файл, добавляет элемент в список,
        /// затем полностью пересохраняет файл.
        /// </para>
        /// </summary>
        /// <param name="path">Путь к JSON-файлу с примерами.</param>
        /// <param name="sample">Добавляемый пример.</param>
        public static void Append(string path, Sample sample)
        {
            if (sample == null) throw new ArgumentNullException(nameof(sample));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Bad path", nameof(path));

            var list = Load(path);
            list.Add(sample);
            SaveAll(path, list);
        }

        /// <summary>
        /// Загружает все примеры из JSON-файла.
        /// </summary>
        /// <param name="path">Путь к JSON-файлу с примерами.</param>
        /// <returns>
        /// Список примеров. Если файл не существует — возвращается пустой список.
        /// Некорректные элементы (например, с <c>X == null</c>) пропускаются.
        /// </returns>
        public static List<Sample> Load(string path)
        {
            if (!File.Exists(path))
                return new List<Sample>();

            var json = File.ReadAllText(path);

            // Если JSON пустой/битый/не того формата, десериализация может вернуть null,
            // поэтому подстраховываемся пустым списком.
            var dtos = JsonSerializer.Deserialize<List<SampleDto>>(json) ?? new List<SampleDto>();

            var list = new List<Sample>(dtos.Count);
            foreach (var d in dtos)
            {
                if (d.X == null) continue;
                list.Add(new Sample(d.X, d.Label));
            }

            return list;
        }

        /// <summary>
        /// Полностью перезаписывает файл примеров указанным списком.
        /// <para>
        /// Создаёт директорию под файл, если её нет.
        /// JSON сохраняется с отступами (WriteIndented) для удобства чтения.
        /// </para>
        /// </summary>
        /// <param name="path">Путь к JSON-файлу с примерами.</param>
        /// <param name="samples">Список примеров, который нужно сохранить.</param>
        public static void SaveAll(string path, List<Sample> samples)
        {
            var dtos = new List<SampleDto>(samples.Count);
            foreach (var s in samples)
            {
                // Важно: X сохраняется как есть (ссылка на массив).
                // Если требуется "заморозка" данных на момент сохранения — можно сохранять копию.
                dtos.Add(new SampleDto { Label = s.Label, X = s.X });
            }

            var opt = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(dtos, opt);

            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllText(path, json);
        }
    }
}