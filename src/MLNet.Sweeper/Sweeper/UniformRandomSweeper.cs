// <copyright file="UniformRandomSweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System.Linq;
using Microsoft.ML.Runtime;

namespace MLNet.Sweeper
{
    /// <summary>
    /// Random sweeper, it generates random values for each of the parameters.
    /// </summary>
    public sealed class UniformRandomSweeper : SweeperBase
    {
        public UniformRandomSweeper(IHostEnvironment env, SweeperOptionBase options)
            : base(options, env, "UniformRandom")
        {
        }

        public UniformRandomSweeper(IHostEnvironment env, SweeperOptionBase options, IValueGenerator[] sweepParameters)
            : base(options, env, sweepParameters, "UniformRandom")
        {
        }

        protected override ParameterSet CreateParamSet()
        {
            return new ParameterSet(this.SweepParameters.Select(sweepParameter => sweepParameter.CreateFromNormalized(this.Host.Rand.NextDouble())));
        }
    }
}
