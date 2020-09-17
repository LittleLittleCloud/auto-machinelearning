// <copyright file="SerializableEstimatorCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.Expert.Extension;

namespace MLNet.Expert
{
    internal class SerializableEstimatorCatalog
    {
        private MLContext context;

        public SerializableEstimatorCatalog(MLContext context)
        {
            this.context = context;
            this.BinaryClassification = new SerializableBinaryClassificationTrainerCatalog(context);
            this.MultiClassification = new SerializableMultiClassificationTrainerCatalog(context);
            this.Regression = new SerializableRegressionTrainerCatalog(context);
            this.Transformer = new SerializableTransformerCatalog(context);
        }

        public SerializableBinaryClassificationTrainerCatalog BinaryClassification { get; private set; }

        public SerializableMultiClassificationTrainerCatalog MultiClassification { get; private set; }

        public SerializableRegressionTrainerCatalog Regression { get; private set; }

        public SerializableTransformerCatalog Transformer { get; private set; }
    }
}