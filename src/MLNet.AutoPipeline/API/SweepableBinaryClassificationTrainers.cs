// <copyright file="SweepableBinaryClassificationTrainers.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// Class for all sweepable binary classification trainers.
    /// </summary>
    public class SweepableBinaryClassificationTrainers
    {
        internal SweepableBinaryClassificationTrainers(MLContext context)
        {
            this.Context = context;
        }

        internal MLContext Context { get; private set; }
    }
}