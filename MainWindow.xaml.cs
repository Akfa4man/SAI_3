using SAI_3.Data;
using SAI_3.Neural;
using System.Globalization;
using System.Windows;
using System.Windows.Controls.Primitives;
using System.Windows.Media;

namespace SAI_3
{
    /// <summary>
    /// Главное окно WPF-приложения для рисования цифры 5x7, обучения MLP и предсказания.
    /// <para>
    /// Сценарии:
    /// 1) Пользователь кликает по пикселям (ToggleButton) → формируется вектор длиной 35.
    /// 2) Train запускает обучение на синтетическом датасете (DatasetGenerator) в фоне.
    /// 3) Predict делает предсказание по текущему рисунку.
    /// 4) RememberDigit сохраняет пример в samples.json и выполняет быстрое дообучение.
    /// </para>
    /// </summary>
    public partial class MainWindow : Window
    {
        /// <summary>
        /// Состояние пикселей "холста" 5x7, развернутое в одномерный массив длиной 35.
        /// <para>
        /// true  — пиксель закрашен (1.0), false — фон (0.0).
        /// Индексация должна совпадать с Tag кнопок в XAML.
        /// </para>
        /// </summary>
        private readonly bool[] _pixels = new bool[35];

        /// <summary>
        /// Текущая модель нейросети (создаётся при Train/дообучении или загружается из model.json).
        /// </summary>
        private Mlp? _net;

        /// <summary>
        /// Датасет для обучения, генерируется при запуске обучения (Train).
        /// </summary>
        private List<Sample>? _train;

        /// <summary>
        /// Датасет для проверки качества (test), генерируется при запуске обучения (Train).
        /// </summary>
        private List<Sample>? _test;

        /// <summary>
        /// Источник отмены фонового обучения. Пока не null — обучение считается активным.
        /// </summary>
        private CancellationTokenSource? _cts;

        /// <summary>
        /// Путь к файлу модели (JSON) в директории приложения.
        /// </summary>
        private readonly string _modelPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model.json");

        /// <summary>
        /// Путь к файлу пользовательских примеров (JSON) в директории приложения.
        /// </summary>
        private readonly string _samplesPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "samples.json");

        /// <summary>
        /// Признак, что обучение запущено (есть активный CTS).
        /// </summary>
        private bool IsTraining => _cts != null;

        /// <summary>
        /// Признак, что модель доступна для предсказания/сохранения.
        /// </summary>
        private bool HasModel => _net != null;

        /// <summary>
        /// Признак, что на холсте есть хотя бы один закрашенный пиксель.
        /// Нужен, чтобы не запускать Predict на пустом рисунке.
        /// </summary>
        private bool HasAnyPixel => Array.Exists(_pixels, p => p);

        /// <summary>
        /// Инициализация окна: подготавливает визуализацию пикселей, синхронизирует состояние UI
        /// и пытается загрузить ранее сохранённую модель из model.json (если файл существует).
        /// </summary>
        public MainWindow()
        {
            InitializeComponent();

            UpdatePixelVisuals();
            UpdateUiState();

            // Автозагрузка модели при старте, если она уже сохранена на диске.
            try
            {
                if (System.IO.File.Exists(_modelPath))
                {
                    _net = ModelIO.Load(_modelPath);
                    StatusText.Text = "Загружена модель из model.json";
                }
            }
            catch (Exception ex)
            {
                // Ошибка загрузки не критична: приложение остаётся работоспособным (можно переобучить).
                StatusText.Text = "Не удалось загрузить model.json: " + ex.Message;
            }
        }

        /// <summary>
        /// Обновляет доступность элементов интерфейса в зависимости от состояния приложения:
        /// обучение/наличие модели/наличие нарисованных пикселей.
        /// </summary>
        private void UpdateUiState()
        {
            TrainButton.IsEnabled = !IsTraining;
            StopButton.IsEnabled = IsTraining;

            PredictButton.IsEnabled = HasModel && HasAnyPixel && !IsTraining;

            RememberPanel.IsEnabled = HasAnyPixel && !IsTraining;

            SaveModelButton.IsEnabled = HasModel && !IsTraining;
            LoadModelButton.IsEnabled = !IsTraining && System.IO.File.Exists(_modelPath);

            ClearButton.IsEnabled = !IsTraining;

            EpochsBox.IsEnabled = !IsTraining;
            BatchSizeBox.IsEnabled = !IsTraining;
            LearningRateBox.IsEnabled = !IsTraining;
            NoiseBox.IsEnabled = !IsTraining;
        }

        /// <summary>
        /// Обработчик клика по пикселю (ToggleButton) на "холсте" 5x7.
        /// <para>
        /// Tag кнопки должен содержать индекс 0..34, соответствующий элементу в <see cref="_pixels"/>.
        /// После изменения обновляется визуальное отображение и состояние UI.
        /// </para>
        /// </summary>
        private void Pixel_Click(object sender, RoutedEventArgs e)
        {
            if (sender is not ToggleButton tb) return;
            if (!int.TryParse(tb.Tag?.ToString(), out int idx)) return;
            if (idx < 0 || idx >= _pixels.Length) return;

            _pixels[idx] = tb.IsChecked == true;
            tb.Background = _pixels[idx] ? Brushes.Black : Brushes.White;

            UpdateUiState();
        }

        /// <summary>
        /// Очищает холст (все пиксели в false) и сбрасывает вывод предсказания.
        /// </summary>
        private void ClearButton_Click(object sender, RoutedEventArgs e)
        {
            Array.Clear(_pixels, 0, _pixels.Length);

            PredText.Text = "";
            ProbsText.Text = "";

            UpdatePixelVisuals();
            UpdateUiState();
        }

        /// <summary>
        /// Синхронизирует визуальные ToggleButton'ы с массивом <see cref="_pixels"/>.
        /// Вызывается при старте и после очистки, чтобы "холст" и UI совпадали.
        /// </summary>
        private void UpdatePixelVisuals()
        {
            foreach (var tb in FindVisualChildren<ToggleButton>(this))
            {
                if (!int.TryParse(tb.Tag?.ToString(), out int idx)) continue;
                if (idx < 0 || idx >= _pixels.Length) continue;

                tb.IsChecked = _pixels[idx];
                tb.Background = _pixels[idx] ? Brushes.Black : Brushes.White;
            }
        }

        /// <summary>
        /// Рекурсивно перечисляет всех визуальных потомков указанного типа в VisualTree.
        /// Используется, чтобы найти все пиксельные ToggleButton'ы без хранения ссылок на них.
        /// </summary>
        private static IEnumerable<T> FindVisualChildren<T>(DependencyObject depObj) where T : DependencyObject
        {
            if (depObj == null) yield break;

            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(depObj); i++)
            {
                var child = VisualTreeHelper.GetChild(depObj, i);
                if (child is T t) yield return t;

                foreach (var childOfChild in FindVisualChildren<T>(child))
                    yield return childOfChild;
            }
        }

        /// <summary>
        /// Делает предсказание для текущего рисунка и выводит:
        /// <list type="bullet">
        /// <item><description>PredText — предсказанный класс (цифра);</description></item>
        /// <item><description>ProbsText — вероятности по всем классам.</description></item>
        /// </list>
        /// </summary>
        private void PredictButton_Click(object sender, RoutedEventArgs e)
        {
            if (IsTraining) return;

            if (_net == null)
            {
                MessageBox.Show("Сначала обучите сеть (Train).", "Нет модели", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // Преобразуем булевы пиксели в double-вектор (0/1) для входа MLP.
            double[] xRaw = _pixels.Select(p => p ? 1.0 : 0.0).ToArray();
            double[] x = Preprocess.CropAndNormalizeTo5x7(xRaw);

            int pred = _net.Predict(x);
            var probs = _net.PredictProba(x);

            PredText.Text = pred.ToString(CultureInfo.InvariantCulture);

            ProbsText.Text = string.Join("  ",
                probs.Select((p, i) => $"{i}:{p:0.000}")
            );
        }

        /// <summary>
        /// Сохраняет текущую модель в model.json.
        /// </summary>
        private void SaveModelButton_Click(object sender, RoutedEventArgs e)
        {
            if (_net == null)
            {
                MessageBox.Show("Нет модели для сохранения.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            try
            {
                ModelIO.Save(_net, _modelPath);
                StatusText.Text = "Модель сохранена в model.json";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Save error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            UpdateUiState();
        }

        /// <summary>
        /// Загружает модель из model.json и заменяет текущую <see cref="_net"/>.
        /// </summary>
        private void LoadModelButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                _net = ModelIO.Load(_modelPath);
                StatusText.Text = "Модель загружена из model.json";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Load error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            UpdateUiState();
        }

        /// <summary>
        /// "Запоминает" текущий рисунок как пример выбранной цифры:
        /// <para>
        /// 1) Добавляет пример в samples.json;</para>
        /// <para>
        /// 2) Если модель ещё не создана — создаёт её;</para>
        /// <para>
        /// 3) Загружает все пользовательские примеры и выполняет дообучение на них;</para>
        /// <para>
        /// 4) Сохраняет обновлённую модель в model.json.</para>
        /// </summary>
        private void RememberDigit_Click(object sender, RoutedEventArgs e)
        {
            if (sender is not System.Windows.Controls.Button btn) return;
            if (!int.TryParse(btn.Content?.ToString(), out int label)) return;

            double[] xRaw = _pixels.Select(p => p ? 1.0 : 0.0).ToArray();
            double[] x = Preprocess.CropAndNormalizeTo5x7(xRaw);

            try
            {
                // Сохраняем пользовательский пример на диск.
                SamplesIO.Append(_samplesPath, new Sample(x, label));
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Samples error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // Если модели нет, создаём новую с фиксированной архитектурой.
            _net ??= new Mlp(inputSize: 35, hiddenSize: 128, outputSize: 10, seed: 42);

            try
            {
                // Дообучаем на всех сохранённых примерах.
                var all = SamplesIO.Load(_samplesPath);

                var rng = new Random(999);
                int epochs = 25;
                int batch = Math.Min(16, Math.Max(1, all.Count));
                double lr = 0.03;

                for (int ep = 0; ep < epochs; ep++)
                    _net.TrainEpoch(all, batch, lr, rng);

                // После дообучения обновляем модель на диске.
                ModelIO.Save(_net, _modelPath);

                StatusText.Text = $"Пример сохранён (label={label}). Дообучено на {all.Count} примерах. Модель обновлена.";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Fine-tune error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            UpdateUiState();
        }

        /// <summary>
        /// Запускает обучение модели на синтетическом датасете в фоновом потоке.
        /// <para>
        /// В UI-потоке делается только подготовка/валидация параметров и обновление прогресса через Dispatcher.
        /// Отмена обучения выполняется через <see cref="_cts"/>.
        /// </para>
        /// </summary>
        private async void TrainButton_Click(object sender, RoutedEventArgs e)
        {
            // Защита от повторного запуска обучения.
            if (_cts != null)
            {
                MessageBox.Show("Обучение уже запущено.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            // Читаем параметры обучения из UI (epochs, batchSize, lr, noise).
            if (!TryReadTrainingParams(out int epochs, out int batchSize, out double lr, out double noise))
                return;

            // Генерируем синтетические данные (шаблоны цифр + шум).
            _train = DatasetGenerator.Generate(perClass: 400, noiseFlipProb: noise, seed: 123, shuffle: true);
            _test = DatasetGenerator.Generate(perClass: 60, noiseFlipProb: noise, seed: 456, shuffle: true);

            // Создаём новую модель (архитектура фиксирована для обучения "с нуля").
            _net = new Mlp(inputSize: 35, hiddenSize: 64, outputSize: 10, seed: 42);

            // Немедленная блокировка кнопок — чтобы пользователь видел, что процесс стартовал.
            TrainButton.IsEnabled = false;
            StopButton.IsEnabled = true;
            TrainProgress.Value = 0;
            StatusText.Text = "Старт обучения...";

            _cts = new CancellationTokenSource();
            var token = _cts.Token;

            UpdateUiState();

            try
            {
                // Запуск обучения в пуле потоков. UI не блокируется.
                await Task.Run(() => TrainLoop(_net, _train, _test, epochs, batchSize, lr, token), token);
            }
            catch (OperationCanceledException)
            {
                // Отмена — ожидаемый сценарий (например, пользователь нажал Stop).
                StatusText.Text = "Обучение остановлено пользователем.";
            }
            catch (Exception ex)
            {
                StatusText.Text = "Ошибка обучения: " + ex.Message;
            }
            finally
            {
                // Освобождаем CTS и возвращаем UI в "необучающее" состояние.
                _cts?.Dispose();
                _cts = null;

                TrainButton.IsEnabled = true;
                StopButton.IsEnabled = false;

                UpdateUiState();
            }
        }

        /// <summary>
        /// Запрашивает остановку обучения.
        /// <para>
        /// Фактическая остановка происходит в <see cref="TrainLoop"/> при проверке token / IsCancellationRequested.
        /// </para>
        /// </summary>
        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            _cts?.Cancel();
            UpdateUiState();
        }

        /// <summary>
        /// Цикл обучения (выполняется в фоне).
        /// <para>
        /// На каждой эпохе: обучаемся, считаем loss/accuracy и обновляем UI через Dispatcher.
        /// Отмена: при запросе отмены цикл досрочно завершает обучение.
        /// </para>
        /// </summary>
        /// <param name="net">Обучаемая сеть.</param>
        /// <param name="train">Обучающий датасет.</param>
        /// <param name="test">Тестовый датасет.</param>
        /// <param name="epochs">Число эпох.</param>
        /// <param name="batchSize">Размер батча.</param>
        /// <param name="lr">Шаг обучения.</param>
        /// <param name="token">Токен отмены обучения.</param>
        private void TrainLoop(Mlp net, List<Sample> train, List<Sample> test,
            int epochs, int batchSize, double lr, CancellationToken token)
        {
            var rng = new Random(777);

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                // Проверяем отмену "мягко": выходим из цикла без исключения.
                if (token.IsCancellationRequested)
                    break;

                double loss = net.TrainEpoch(train, batchSize, lr, rng);
                double trainAcc = net.EvaluateAccuracy(train);
                double testAcc = net.EvaluateAccuracy(test);

                int progress = (int)Math.Round(epoch * 100.0 / epochs);

                // UI обновляем только через Dispatcher, т.к. TrainLoop выполняется не в UI-потоке.
                Dispatcher.Invoke(() =>
                {
                    TrainProgress.Value = progress;
                    StatusText.Text =
                        $"Epoch {epoch}/{epochs}  |  loss={loss:0.0000}  |  trainAcc={trainAcc * 100:0.00}%  |  testAcc={testAcc * 100:0.00}%";
                });
            }

            Dispatcher.Invoke(() =>
            {
                StatusText.Text += "\nГотово. Можно рисовать цифру и нажимать Predict.";
            });
        }

        /// <summary>
        /// Считывает параметры обучения из текстовых полей UI и валидирует их.
        /// <para>
        /// В случае ошибки показывает MessageBox и возвращает false.
        /// </para>
        /// </summary>
        /// <param name="epochs">Число эпох (&gt; 0).</param>
        /// <param name="batchSize">Размер батча (&gt; 0).</param>
        /// <param name="lr">Шаг обучения (&gt; 0), парсится в InvariantCulture.</param>
        /// <param name="noise">Вероятность инверсии пикселя [0..1], парсится в InvariantCulture.</param>
        /// <returns>true, если все параметры корректны; иначе false.</returns>
        private bool TryReadTrainingParams(out int epochs, out int batchSize, out double lr, out double noise)
        {
            epochs = 0; batchSize = 0; lr = 0; noise = 0;

            if (!int.TryParse(EpochsBox.Text, out epochs) || epochs <= 0)
            {
                MessageBox.Show("Epochs должно быть целым > 0", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            if (!int.TryParse(BatchSizeBox.Text, out batchSize) || batchSize <= 0)
            {
                MessageBox.Show("BatchSize должно быть целым > 0", "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            if (!double.TryParse(LearningRateBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out lr) || lr <= 0)
            {
                MessageBox.Show("LearningRate должно быть числом > 0 (используй точку, например 0.05)", "Ошибка",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            if (!double.TryParse(NoiseBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out noise) || noise < 0 || noise > 1)
            {
                MessageBox.Show("NoiseFlipProb должно быть числом в диапазоне [0..1]", "Ошибка",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }

            return true;
        }
    }
}