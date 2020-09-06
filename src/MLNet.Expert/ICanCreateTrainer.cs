// <copyright file="ICanCreateTrainer.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.AutoPipeline;

namespace MLNet.Expert
{
    public interface ICanCreateTrainer
    {
        /// <summary>
        /// Create a ml.net trainer using <paramref name="input"/> as input and save output to <paramref name="output"/>.
        /// </summary>
        /// <param name="context">ml context.</param>
        /// <param name="label">label column name.</param>
        /// <param name="feature">feature column name.</param>
        /// <returns></returns>
        SweepableEstimatorBase CreateTrainer(MLContext context, string label, string feature);
    }
}
