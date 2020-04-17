// <copyright file="ISweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.Sweeper;
using System.Collections.Generic;

namespace MLNet.AutoPipeline
{
    internal interface ISweeper : IEnumerable<ParameterSet>, IEnumerator<ParameterSet>
    {
        /// <summary>
        /// For trainable Sweeper.
        /// </summary>
        /// <param name="input">Output of Sweeper.</param>
        /// <param name="Y">Score from model</param>
        void AddRunHistory(IRunResult input);
    }

    internal class SweeperInput
    {
        public double Score { get; set; }
    }
}
