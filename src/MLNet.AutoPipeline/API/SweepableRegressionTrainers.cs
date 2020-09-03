// <copyright file="SweepableRegressionTrainers.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// class for sweepable regression trainers.
    /// </summary>
    public class SweepableRegressionTrainers
    {
        internal SweepableRegressionTrainers(MLContext context)
        {
            this.Context = context;
        }

        internal MLContext Context { get; private set; }
    }
}