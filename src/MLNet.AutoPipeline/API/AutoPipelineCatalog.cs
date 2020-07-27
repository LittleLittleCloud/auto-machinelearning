// <copyright file="AutoPipelineCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// A catalog for all avalilable AutoPipeline API.
    /// </summary>
    public sealed class AutoPipelineCatalog
    {
        private MLContext mlContext;

        public AutoPipelineCatalog(MLContext context)
        {
            this.mlContext = context;
            this.MultiClassification = new SweepableMultiClassificationTrainers(context);
            this.BinaryClassification = new SweepableBinaryClassificationTrainers(context);
        }

        public SweepableBinaryClassificationTrainers BinaryClassification { get; private set; }

        public SweepableMultiClassificationTrainers MultiClassification { get; private set; }

        public SweepableRegressionTrainers Regression { get; private set; }

    }
}
