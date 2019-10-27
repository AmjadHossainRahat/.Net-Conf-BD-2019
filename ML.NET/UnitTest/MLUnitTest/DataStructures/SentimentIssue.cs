using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataStructures
{
    public class SentimentIssue
    {
        [LoadColumn(0)]
        public bool Label { get; set; }
        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
