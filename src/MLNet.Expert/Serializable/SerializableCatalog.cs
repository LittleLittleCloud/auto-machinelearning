// <copyright file="SerializableCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.Expert.Serializable;

namespace MLNet.Expert
{
    internal class SerializableCatalog
    {
        private MLContext context;

        public SerializableCatalog(MLContext context)
        {
            this.context = context;
            this.BinaryClassification = new SerializableBinaryClassificationTrainerCatalog(context);
            this.MultiClassification = new SerializableMultiClassificationTrainerCatalog(context);
            this.Regression = new SerializableRegressionTrainerCatalog(context);
            this.Transformer = new SerializableTransformerCatalog(context);
            this.Sweeper = new SerializableSweeperCatalog(context);
            this.EvaluationFunction = new SerializableEvaluateFunction(context);
            this.Factory = new FactoryCatalog(context);
        }

        public SerializableBinaryClassificationTrainerCatalog BinaryClassification { get; }

        public SerializableMultiClassificationTrainerCatalog MultiClassification { get; }

        public SerializableRegressionTrainerCatalog Regression { get; }

        public SerializableTransformerCatalog Transformer { get; }

        public SerializableSweeperCatalog Sweeper { get; }

        public SerializableEvaluateFunction EvaluationFunction { get; }

        public FactoryCatalog Factory { get; }
    }
}