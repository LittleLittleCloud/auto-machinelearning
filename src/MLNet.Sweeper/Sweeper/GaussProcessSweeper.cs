// <copyright file="GaussProcessSweeper.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.Sweeper
{
    public class GaussProcessSweeper : ISweeper
    {
        private ISweeper _randomSweeper;
        private GaussProcessRegressor _regressor;
        private HashSet<IDictionary<string, string>> _generated;
        private Option _option;
        private IList<IRunResult> _runHistory;
        private Random _rand = new Random();
        private IEnumerable<IValueGenerator> _valueGenerators;

        public Parameters Current { get; private set; }

        public GaussProcessSweeper(Option option)
        {
            this._option = option;
            this._generated = new HashSet<IDictionary<string, string>>();
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

        public IEnumerable<IDictionary<string, string>> ProposeSweeps(ISweepable sweepable, int maxSweeps = 100, IEnumerable<IRunResult> previousRuns = null)
        {
            this._valueGenerators = sweepable.SweepableValueGenerators;
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
                    var randomSweepResult = this._randomSweeper.ProposeSweeps(sweepable, 1, this._runHistory).First();
                    this._generated.Add(randomSweepResult);
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
                    var candidates = new List<IDictionary<string, string>>();
                    foreach ( var parent in kBestParents)
                    {
                        var parameter = new Parameters(parent.ParameterSet.Select(kv =>
                        {
                            var generator = this._valueGenerators.Where(g => g.Name == kv.Key).FirstOrDefault();
                            return generator.CreateFromString(kv.Value);
                        }));
                        var _candidates = this.GetOneMutationNeighbourhood(parameter);
                        candidates.AddRange(_candidates);
                    }

                    // add some random samples
                    var random_sample = this._randomSweeper.ProposeSweeps(sweepable, 20, this._runHistory).ToList();
                    candidates.AddRange(random_sample);

                    // prepare train data
                    var X_train = this.CreateNDarrayFromParamaters(this._runHistory.Select(history => history.ParameterSet));
                    var y_train = np.array(this._runHistory.Select(x => x.IsMetricMaximizing ? (double)x.MetricValue : -(double)x.MetricValue).ToArray()).reshape(-1, 1);
                    var X_sample = this.CreateNDarrayFromParamaters(candidates);

                    // fit
                    (var predict_y, var std, var cov) = this._regressor.Fit(X_train, y_train).Transform(X_sample);

                    // calulate ei
                    var ei = GaussProcessSweeper.EI(predict_y.reshape(-1, 1), std.reshape(-1, 1), np.max(y_train));
                    var bestCandidate = (int)np.argmax(ei);

                    if ((double)np.max(ei) <= 0)
                    {
                        bestCandidate = this._rand.Next(0, ei.len);
                    }

                    this._generated.Add(candidates[bestCandidate]);
                    yield return candidates[bestCandidate];
                }
            }
        }

        private NDarray CreateNDarrayFromParamaters(IEnumerable<IDictionary<string, string>> parameters)
        {
            NDarray res = null;
            foreach ( var paramterSet in parameters)
            {
                var doubleArray = new List<double>();
                foreach (var paramater in paramterSet)
                {
                    doubleArray.Add(double.Parse(paramater.Value));
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

        public static NDarray EI(NDarray predictY, NDarray std, NDarray bestY, double xi = 0.01)
        {
            std = std.reshape(-1, 1);
            var imp = predictY - bestY - (NDarray)xi;
            var Z = imp / std;
            var ei = imp * Utils.NormCDF(Z) + std * Utils.NormPDF(Z);
            ei[std < (NDarray)0] = (NDarray)0;
            return ei;
        }

        public object Clone()
        {
            return new GaussProcessSweeper(this._option);
        }

        private IDictionary<string, string>[] GetOneMutationNeighbourhood(Parameters parent)
        {
            var candicates = new List<Parameters>();
            foreach (var valueGenerator in this._valueGenerators)
            {
                var _candidates = parent.Clone();
                var value = _candidates[valueGenerator.ID];
                if (valueGenerator is INumericValueGenerator)
                {
                    var norm = (valueGenerator as INumericValueGenerator).NormalizeValue(value);
                    var next = 1e-2 * Utils.Normal() + norm;
                    while (true)
                    {
                        if (next > 0 || next < 1)
                        {
                            break;
                        }

                        next = 1e-2 * Utils.Normal() + norm;
                    }

                    _candidates[valueGenerator.ID] = valueGenerator.CreateFromNormalized(next);
                }
                else
                {
                    _candidates[valueGenerator.ID] = (valueGenerator as IDiscreteValueGenerator).CreateFromNormalized(this._rand.NextDouble());
                }

                candicates.Add(_candidates);
            }

            return candicates.Select(x => x.ParameterValues).ToArray();
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
