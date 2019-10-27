using DataStructures;
using Microsoft.ML;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLUnitTest.Test
{
    [TestClass]
    class MLModelAccuracyTest
    {
        private static readonly string BaseDatasetsRelativePath = @"../../../../Data";
        private static readonly string DataRelativePath = $"{BaseDatasetsRelativePath}/sentiment-analysis-wiki-250-line-data.tsv";
        private static readonly string DataPath = GetAbsolutePath(DataRelativePath);


        // Just a naive sample how you can test accuracy of your trained model with Test Data set
        [TestMethod]
        public void TestAccuracyHigherThan75()
        {
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

            double accuracy = metrics.Accuracy;
            Console.WriteLine($"Accuracy of model in this validation '{accuracy * 100}'%");

            Assert.IsTrue(accuracy > 0.75, "Accuracy was not greater than 0.75");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            /// Ofcourse this is not how you do this in real life!!
            FileInfo _dataRoot = new FileInfo(typeof(MLUnitTest.Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
