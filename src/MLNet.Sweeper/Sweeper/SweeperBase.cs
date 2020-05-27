// <copyright file="SweeperBase.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;

namespace MLNet.Sweeper
{
    /// <summary>
    /// Base sweeper that ensures the suggestions are different from each other and from the previous runs.
    /// </summary>
    public abstract class SweeperBase : ISweeper
    {
        private HashSet<ParameterSet> _generated;
        protected IList<IRunResult> _history = new List<IRunResult>();

        private readonly SweeperOptionBase _options;

        public ParameterSet Current { get; private set; }

        public IEnumerable<IValueGenerator> SweepableParamaters { get; set; }

        protected SweeperBase(SweeperOptionBase options, string name)
        {
            this._options = options;
            this.SweepableParamaters = new List<IValueGenerator>();
            this._generated = new HashSet<ParameterSet>();
        }

        protected SweeperBase(SweeperOptionBase options, IValueGenerator[] sweepParameters, string name)
        {
            this._options = options;
            this.SweepableParamaters = sweepParameters;
        }

        public virtual IEnumerable<ParameterSet> ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            ParameterSet candidate;

            if (previousRuns != null)
            {
                this._history.Concat(previousRuns);
            }

            for (int i = 0; i != maxSweeps; ++i)
            {
                var retry = 0;
                while (true)
                {
                    retry++;
                    candidate = this.CreateParamSet();
                    if (retry >= this._options.Retry ||
                        !AlreadyGenerated(candidate, this._history.Select(x => x.ParameterSet)) ||
                        !this._generated.Contains(candidate))
                    {
                        break;
                    }
                }

                this._generated.Add(candidate);
                this.Current = candidate;
                yield return candidate;
            }
        }

        protected static bool AlreadyGenerated(ParameterSet paramSet, IEnumerable<ParameterSet> previousRuns)
        {
            return previousRuns.Any(previousRun => previousRun.Equals(paramSet));
        }

        protected abstract ParameterSet CreateParamSet();

        public virtual void AddRunHistory(IRunResult input)
        {
            this._history.Add(input);
        }

        public abstract object Clone();
    }
}
