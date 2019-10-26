using DataStructures;
using Microsoft.ML;
using Newtonsoft.Json;
using System;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace BasicTest
{
    class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"../../../../Data";
        private static readonly string DataRelativePath = $"{BaseDatasetsRelativePath}/sentiment-analysis-wiki-250-line-data.tsv";
        private static readonly string DataPath = GetAbsolutePath(DataRelativePath);

        private static readonly string BaseModelsRelativePath = @"../../../../MLModels";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/SentimentModel.zip";
        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            Console.WriteLine("Hello ML World!");

            var mlContext = new MLContext();

            // STEP 1: Common data loading configuration
            IDataView dataView = mlContext.Data
                                .LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);

            // 0.2 testFraction means: (Training Data: Test Data) = (80 : 20)
            TrainTestData trainTestSplit = mlContext.Data
                                            .TrainTestSplit(dataView, testFraction: 0.2);

            IDataView trainingData = trainTestSplit.TrainSet;
            IDataView testData = trainTestSplit.TestSet;

            // STEP 2: Common data process configuration with pipeline data transformations          
            var dataProcessPipeline = mlContext.Transforms
                .Text.FeaturizeText(outputColumnName: "Features", 
                inputColumnName: nameof(SentimentIssue.Text));

            // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
            var trainer = mlContext.BinaryClassification
                .Trainers.SdcaLogisticRegression();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);

            // STEP 5: Evaluate the model and show accuracy stats
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions);

            Console.WriteLine($"Evaluation result: {JsonConvert.SerializeObject(metrics)}");

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);

            Console.WriteLine($"The model is saved to {ModelPath}");

            // STEP 6: Create prediction engine
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

            // STEP 6: Make a prediction
            var prediction = predictionEngine.Predict(new SentimentIssue { Text = "I am testing ML and it looks good" });

            Console.WriteLine($"Prediction: {JsonConvert.SerializeObject(prediction)}");
            Console.ReadKey();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
