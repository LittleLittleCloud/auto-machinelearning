// <copyright file="SweepableMultiClassificationTrainers.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// class for sweepable multi classification trainers.
    /// </summary>
    public class SweepableMultiClassificationTrainers
    {
        internal SweepableMultiClassificationTrainers(MLContext context)
        {
            this.Context = context;
        }

        internal MLContext Context { get; private set; }
    }
}