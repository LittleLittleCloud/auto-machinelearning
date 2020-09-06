// <copyright file="ITransformExpert.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.AutoPipeline;
using System.Collections.Generic;

namespace MLNet.Expert
{
    /// <summary>
    /// Interface for all Transform Expert classes.
    /// </summary>
    public interface ITransformExpert
    {
        /// <summary>
        /// Propose <see cref="IEstimatorNode"/> that will be applied on <paramref name="inputColumn"/> and result will be saved in <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="inputColumn">input column name.</param>
        /// <param name="outputColumn">output column name.</param>
        /// <returns>IEstimatorNode.</returns>
        IEnumerable<SweepableEstimatorBase> Propose(string inputColumn, string outputColumn);

        /// <summary>
        /// Propose <see cref="IEstimatorNode"/> that will be applied on <paramref name="inputColumn"/> and result will be saved in-place.
        /// </summary>
        /// <param name="inputColumn">input column name.</param>
        /// <returns>IEstimatorNode.</returns>
        IEnumerable<SweepableEstimatorBase> Propose(string inputColumn);
    }
}
