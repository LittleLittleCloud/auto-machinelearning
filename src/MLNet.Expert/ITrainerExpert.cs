// <copyright file="ITrainerExpert.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.AutoPipeline;

namespace MLNet.Expert
{
    /// <summary>
    /// Interface for all trainer expert class.
    /// </summary>
    public interface ITrainerExpert
    {
        /// <summary>
        /// Propose a group of trainers that can be applied in pipeline.
        /// </summary>
        /// <param name="label">label column name.</param>
        /// <param name="feature">feature column name.</param>
        /// <returns></returns>
        IEstimatorNode Propose(string label, string feature);
    }
}
