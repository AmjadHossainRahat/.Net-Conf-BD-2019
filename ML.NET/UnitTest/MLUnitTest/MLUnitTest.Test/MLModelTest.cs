using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLUnitTest.Model;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLUnitTest.Test
{
    [TestClass]
    class MLModelTest
    {
        [TestMethod]
        public void TestPositiveSentimentStatement()
        {
            // Add input data
            var input = new ModelInput();
            input.SentimentText = "ML.NET is awesome!";

            // Load model and predict output of sample data
            ModelOutput result = ConsumeModel.Predict(input);

            Assert.AreEqual(true, Convert.ToBoolean(result.Prediction));
        }

        [TestMethod]
        public void TestNegativeSentimentStatement()
        {
            // Add input data
            var input = new ModelInput();
            input.SentimentText = "The day is so boring!";

            // Load model and predict output of sample data
            ModelOutput result = ConsumeModel.Predict(input);

            Assert.AreEqual(false, Convert.ToBoolean(result.Prediction));
        }
    }
}
