// <copyright file="SweeperBase.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Runtime;
using System.Collections.Generic;
using System.Linq;

namespace MLNet.Sweeper
{
    /// <summary>
    /// Base sweeper that ensures the suggestions are different from each other and from the previous runs.
    /// </summary>
    public abstract class SweeperBase : ISweeper
    {
        protected readonly IValueGenerator[] SweepParameters;
        protected readonly IHost Host;

        private readonly OptionsBase _options;

        protected SweeperBase(OptionsBase options, IHostEnvironment env, string name)
        {
            this.Host = env.Register(name);
            this._options = options;

            this.SweepParameters = options.SweptParameters.Select(p => p.CreateComponent(this.Host)).ToArray();
        }

        protected SweeperBase(OptionsBase options, IHostEnvironment env, IValueGenerator[] sweepParameters, string name)
        {
            this.Host = env.Register(name);
            this._options = options;
            this.SweepParameters = sweepParameters;
        }

        public virtual ParameterSet[] ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            var prevParamSets = previousRuns?.Select(r => r.ParameterSet).ToList() ?? new List<ParameterSet>();
            var result = new HashSet<ParameterSet>();
            for (int i = 0; i < maxSweeps; i++)
            {
                ParameterSet paramSet;
                int retries = 0;
                do
                {
                    paramSet = this.CreateParamSet();
                    ++retries;
                }
                while (paramSet != null && retries < this._options.Retries &&
                    (AlreadyGenerated(paramSet, prevParamSets) || AlreadyGenerated(paramSet, result)));

                result.Add(paramSet);
            }

            return result.ToArray();
        }

        protected static bool AlreadyGenerated(ParameterSet paramSet, IEnumerable<ParameterSet> previousRuns)
        {
            return previousRuns.Any(previousRun => previousRun.Equals(paramSet));
        }

        protected abstract ParameterSet CreateParamSet();
    }
}
