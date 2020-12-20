using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.NNI
{
    public class Iris
    {
        [LoadColumn(0)]
        public float sepal_length;

        [LoadColumn(1)]
        public float sepal_width;

        [LoadColumn(2)]
        public float petal_length;

        [LoadColumn(3)]
        public float petal_width;

        [LoadColumn(4)]
        public string species;
    }
}
