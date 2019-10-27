using AutoMLTestML.Model;
using System;

namespace AutoMLTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello ML.NET World!");

            // Add input data
            var input = new ModelInput();
            input.SentimentText = "ML.net is good but not the best";

            // Load model and predict output of sample data
            ModelOutput result = ConsumeModel.Predict(input);

            Console.WriteLine($"Prediction: {result.Prediction}, Score: {result.Score}");
            Console.ReadKey();
        }
    }
}
