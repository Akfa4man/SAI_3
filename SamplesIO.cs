using SAI_3.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace SAI_3
{
    public static class SamplesIO
    {
        private sealed class SampleDto
        {
            public int Label { get; set; }
            public double[] X { get; set; } = Array.Empty<double>();
        }

        public static void Append(string path, Sample sample)
        {
            if (sample == null) throw new ArgumentNullException(nameof(sample));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Bad path", nameof(path));

            var list = Load(path);
            list.Add(sample);
            SaveAll(path, list);
        }

        public static List<Sample> Load(string path)
        {
            if (!File.Exists(path))
                return new List<Sample>();

            var json = File.ReadAllText(path);
            var dtos = JsonSerializer.Deserialize<List<SampleDto>>(json) ?? new List<SampleDto>();

            var list = new List<Sample>(dtos.Count);
            foreach (var d in dtos)
            {
                if (d.X == null) continue;
                list.Add(new Sample(d.X, d.Label));
            }
            return list;
        }

        public static void SaveAll(string path, List<Sample> samples)
        {
            var dtos = new List<SampleDto>(samples.Count);
            foreach (var s in samples)
            {
                dtos.Add(new SampleDto { Label = s.Label, X = s.X });
            }

            var opt = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(dtos, opt);

            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllText(path, json);
        }
    }
}
