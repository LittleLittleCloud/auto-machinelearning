// <copyright file="AutoPipelineCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// A catalog for all avalilable AutoPipeline API.
    /// </summary>
    public sealed class AutoPipelineCatalog
    {
        private MLContext mlContext;

        public AutoPipelineCatalog(MLContext context)
        {
            this.mlContext = context;
            this.MultiClassification = new SweepableMultiClassificationTrainers(context);
            this.BinaryClassification = new SweepableBinaryClassificationTrainers(context);
        }

        public SweepableBinaryClassificationTrainers BinaryClassification { get; private set; }

        public SweepableMultiClassificationTrainers MultiClassification { get; private set; }

        public SweepableRegressionTrainers Regression { get; private set; }

        /// <summary>
        /// Create a sweepable mlnet trainer using custom factory method that can be used in <see cref="SweepablePipeline"/>.
        /// </summary>
        /// <typeparam name="TTrain">type of trainer, must be <see cref="IEstimator{TTransformer}"/>.</typeparam>
        /// <typeparam name="TOption">option class.</typeparam>
        /// <param name="trainerFactory">factory method that creates the trainer.</param>
        /// <param name="optionBuilder">option builder.</param>
        /// <param name="inputs">input column names.</param>
        /// <param name="output">output column name.</param>
        /// <param name="trainerName">trainer name.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public SweepableNode<TTrain, TOption>
            SweepableTrainer<TTrain, TOption>(Func<MLContext, TOption, TTrain> trainerFactory, SweepableOption<TOption> optionBuilder, string[] inputs = null, string output = null, string trainerName = null)
            where TTrain : IEstimator<ITransformer>
            where TOption : class
        {
            Func<TOption, TTrain> factory = (option) =>
            {
                return trainerFactory(this.mlContext, option);
            };

            return Util.CreateSweepableNode(factory, optionBuilder, estimatorName: trainerName, inputs: inputs, outputs: new string[] { output });
        }

        public Experiment CreateExperiment(SweepablePipeline pipeline, Experiment.Option option = null)
        {
            if (option == null)
            {
                option = new Experiment.Option();
            }

            return new Experiment(this.mlContext, pipeline, option);
        }
    }
}
