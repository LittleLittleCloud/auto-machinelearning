// <copyright file="Util.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline;
using MLNet.Expert.Contract;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;

namespace MLNet.Expert
{
    internal static class Util
    {
        private static Random rng = new Random();

        public static SweepableEstimator<TNewTrain, TOption> CreateSweepableNode<TNewTrain, TOption>(Func<TOption, TNewTrain> estimatorFactory, SweepableOption<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything, string estimatorName = null)
            where TNewTrain : IEstimator<ITransformer>
            where TOption : class
        {
            return new SweepableEstimator<TNewTrain, TOption>(estimatorFactory, optionBuilder, scope, estimatorName);
        }

        public static SweepableEstimator<TInstance> CreateUnSweepableNode<TInstance>(TInstance instance, TransformerScope scope = TransformerScope.Everything, string estimatorName = null)
            where TInstance : IEstimator<ITransformer>
        {
            return new SweepableEstimator<TInstance>(instance, scope, estimatorName);
        }

        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static IEnumerable<T> PickN<T>(this IEnumerable<T> list, int n)
        {
            var pickIndex = Enumerable.Range(0, list.Count()).ToList();
            pickIndex.Shuffle();
            return pickIndex.GetRange(0, n).Select(i => list.ToArray()[i]);
        }

        public static SweepablePipelineDataContract ToDataContract(this SweepablePipeline pipeline)
        {
            var estimatorContracts = new List<List<SweepableEstimatorDataContract>>();
            var nodes = pipeline.EstimatorGenerators;

            foreach (var node in nodes)
            {
                var estimators = new List<SweepableEstimatorDataContract>();
                for (int i = 0; i != node.Count; ++i)
                {
                    var estimator = node[i].RawValue as SweepableEstimatorBase;
                    var estimatorContract = new SweepableEstimatorDataContract()
                    {
                        EstimatorName = estimator.EstimatorName,
                        InputColumns = estimator.InputColumns,
                        OutputColumns = estimator.OutputColumns,
                        Scope = estimator.Scope,
                    };
                    estimators.Add(estimatorContract);
                }

                estimatorContracts.Add(estimators);
            }

            return new SweepablePipelineDataContract()
            {
                Estimators = estimatorContracts,
            };
        }

        public static SingleEstimatorSweepablePipelineDataContract ToDataContract(this SingleEstimatorSweepablePipeline pipeline)
        {
            var estimatorContracts = new List<SweepableEstimatorDataContract>();
            var nodes = pipeline.Estimators;

            foreach (var node in nodes)
            {
                var estimatorContract = new SweepableEstimatorDataContract()
                {
                    EstimatorName = node.EstimatorName,
                    InputColumns = node.InputColumns,
                    OutputColumns = node.OutputColumns,
                    Scope = node.Scope,
                };

                estimatorContracts.Add(estimatorContract);
            }

            return new SingleEstimatorSweepablePipelineDataContract()
            {
                Estimators = estimatorContracts,
            };
        }

        public static SweepablePipeline ToPipeline(this SweepablePipelineDataContract pipelineContract, MLContext context)
        {
            var sweepablePipeline = new SweepablePipeline();

            foreach (var node in pipelineContract.Estimators)
            {
                sweepablePipeline.Append(node.Select(n => context.AutoML().Serializable().Factory.CreateSweepableEstimator(n)).ToArray());
            }

            return sweepablePipeline;
        }

        public static SingleEstimatorSweepablePipeline ToPipeline(this SingleEstimatorSweepablePipelineDataContract pipelineContract, MLContext context)
        {
            var estimators = new List<SweepableEstimatorBase>();
            foreach (var estimator in pipelineContract.Estimators)
            {
                estimators.Append(context.AutoML().Serializable().Factory.CreateSweepableEstimator(estimator));
            }

            return new SingleEstimatorSweepablePipeline(estimators);
        }
    }
}
