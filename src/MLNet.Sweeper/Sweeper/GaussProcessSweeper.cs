// <copyright file="GaussProcessSweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.Sweeper.GP;
using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.Sweeper.Sweeper
{
    public class GaussProcessSweeper : ISweeper
    {
        private ISweeper _randomSweeper;
        private GaussProcessRegressor _regressor;
        private HashSet<ParameterSet> _generated;
        private Option _option;
        private IList<IRunResult> _runHistory;
        private Random _rand = new Random();
        private IEnumerable<IValueGenerator> _valueGenerators;

        public ParameterSet Current { get; private set; }

        public IEnumerable<IValueGenerator> SweepableParamaters
        {
            get
            {
                return this._valueGenerators;
            }

            set
            {
                this._valueGenerators = value;
                this._randomSweeper.SweepableParamaters = value;
            }
        }

        public GaussProcessSweeper(Option option)
        {
            this._option = option;
            this._generated = new HashSet<ParameterSet>();
            this._runHistory = new List<IRunResult>();
            var randomSweeperOption = new UniformRandomSweeper.Option()
            {
                Retry = option.Retry,
            };

            this._randomSweeper = new UniformRandomSweeper(randomSweeperOption);

            var gpOption = new GaussProcessRegressor.Options()
            {
                l = option.L,
                sigma = option.Sigma,
                noise = option.Noise,
            };

            this._regressor = new GaussProcessRegressor(gpOption);
        }

        public void AddRunHistory(IRunResult input)
        {
            this._runHistory.Add(input);
        }

        public IEnumerable<ParameterSet> ProposeSweeps(int maxSweeps, IEnumerable<IRunResult> previousRuns = null)
        {
            if (previousRuns != null)
            {
                foreach ( var history in previousRuns)
                {
                    this._runHistory.Add(history);
                }
            }

            for (int i = 0; i != maxSweeps; ++i)
            {
                if (this._runHistory.Count < this._option.InitialPopulation)
                {
                    var randomSweepResult = this._randomSweeper.ProposeSweeps(1, this._runHistory).First();
                    this._generated.Add(randomSweepResult);
                    this.Current = randomSweepResult;
                    yield return randomSweepResult;
                }
                else
                {
                    // Get K best paramater set
                    IEnumerable<IRunResult> kBestParents;
                    if (this._runHistory.First().IsMetricMaximizing)
                    {
                        kBestParents = this._runHistory.OrderByDescending((result) => result.MetricValue).Take(this._option.KBestParents);
                    }
                    else
                    {
                        kBestParents = this._runHistory.OrderBy((result) => result.MetricValue).Take(this._option.KBestParents);
                    }

                    // Get K*N one-step sample from kBestParents.
                    var candidates = new List<ParameterSet>();
                    foreach ( var parent in kBestParents)
                    {
                        var _candidates = this.GetOneMutationNeighbourhood(parent.ParameterSet);
                        candidates.AddRange(_candidates);
                    }

                    // add some random samples
                    var random_sample = this._randomSweeper.ProposeSweeps(20, this._runHistory).ToList();
                    candidates.AddRange(random_sample);

                    // prepare train data
                    var X_train = this.CreateNDarrayFromParamaterSet(this._runHistory.Select(history => history.ParameterSet));
                    var y_train = np.array(this._runHistory.Select(x => x.IsMetricMaximizing ? (double)x.MetricValue : -(double)x.MetricValue).ToArray()).reshape(-1, 1);
                    var X_sample = this.CreateNDarrayFromParamaterSet(candidates);

                    // fit
                    (var predict_y, var std, var cov) = this._regressor.Fit(X_train, y_train).Transform(X_sample);

                    // calulate ei
                    var ei = GaussProcessSweeper.EI(predict_y.reshape(-1, 1), std.reshape(-1, 1), np.max(y_train));
                    var bestCandidate = (int)np.argmax(ei);
                    Console.WriteLine($"Best candidate: {bestCandidate}");
                    this._generated.Add(candidates[bestCandidate]);
                    this.Current = candidates[bestCandidate];
                    yield return candidates[bestCandidate];
                }
            }
        }

        private NDarray CreateNDarrayFromParamaterSet(IEnumerable<ParameterSet> parameterSets)
        {
            NDarray res = null;
            foreach ( var paramterSet in parameterSets)
            {
                var doubleArray = new List<double>();
                foreach (var paramater in paramterSet)
                {
                    if (paramater is IDiscreteParameterValue)
                    {
                        doubleArray.AddRange((paramater as IDiscreteParameterValue).OneHotEncode);
                    }
                    else
                    {
                        doubleArray.Add(double.Parse(paramater.ValueText));
                    }
                }

                if (res == null)
                {
                    res = np.array(doubleArray.ToArray());
                }
                else
                {
                    res = np.vstack(res, np.array(doubleArray.ToArray()));
                }
            }

            return res;
        }

        public static NDarray EI(NDarray predictY, NDarray std, NDarray bestY, double xi = 0.001)
        {
            std = std.reshape(-1, 1);
            var imp = predictY - bestY - (NDarray)xi;
            var Z = imp / std;
            var ei = imp * Utils.NormCDF(Z) + std * Utils.NormPDF(Z);
            ei[std <= (NDarray)0] = (NDarray)0;
            return ei;
        }

        private ParameterSet[] GetOneMutationNeighbourhood(ParameterSet parent)
        {
            var candicates = new List<ParameterSet>();
            foreach (var valueGenerator in this.SweepableParamaters)
            {
                var _candidates = parent.Clone();
                var value = _candidates[valueGenerator.ID];
                if (valueGenerator is INumericValueGenerator)
                {
                    var norm = (valueGenerator as INumericValueGenerator).NormalizeValue(value);
                    var next = (float)np.random.normal(np.array(norm), np.array(1e-2f));
                    if (next <= 0f || next >= 1f)
                    {
                        next = norm;
                    }

                    _candidates[valueGenerator.ID] = valueGenerator.CreateFromNormalized(next);
                }
                else
                {
                    _candidates[valueGenerator.ID] = (valueGenerator as IDiscreteValueGenerator).CreateFromNormalized(this._rand.NextDouble());
                }

                candicates.Add(_candidates);
            }

            return candicates.ToArray();
        }

        public class Option : SweeperOptionBase
        {
            public double L { get; set; } = 1;

            public double Sigma { get; set; } = 1;

            public double Noise { get; set; } = 1e-5;

            public int InitialPopulation { get; set; } = 20;

            public int KBestParents { get; set; } = 10;
        }
    }
}
