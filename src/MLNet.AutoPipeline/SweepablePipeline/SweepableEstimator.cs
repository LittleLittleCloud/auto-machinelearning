// <copyright file="SweepableEstimator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;
using Newtonsoft.Json;

namespace MLNet.AutoPipeline
{
    public class SweepableEstimator<TTran> : SweepableEstimatorBase
        where TTran : IEstimator<ITransformer>
    {
        private TTran _instance;
        private static SweepableEstimator<IEstimator<ITransformer>> no_op = new SweepableEstimator<IEstimator<ITransformer>>();
        
        public override IEstimator<ITransformer> BuildFromParameters(IDictionary<string, string> parameters)
        {
            return this._instance;
        }

        internal SweepableEstimator(TTran instance, TransformerScope scope = TransformerScope.Everything, string estimatorName = null, string[] inputs = null, string[] outputs = null)
            : base(estimatorName, inputs, outputs, null, scope)
        {
            this._instance = instance;

            if (estimatorName == null)
            {
                this.EstimatorName = instance.ToString().Split('.').Last();
            }
            else
            {
                this.EstimatorName = estimatorName;
            }

            this.InputColumns = inputs;
            this.OutputColumns = outputs;
        }

        private SweepableEstimator() : base(null, null, null, null, TransformerScope.Everything) { }

        internal static SweepableEstimator<IEstimator<ITransformer>> EmptyNode
        {
            get => no_op;
        }

        internal CodeGenNodeContract ToCodeGenNodeContract(Parameters parameters = null)
        {
            return new CodeGenNodeContract()
            {
                EstimatorName = this.EstimatorName,
                InputColumns = this.InputColumns ?? (new string[] { }),
                OutputColumns = this.OutputColumns ?? new string[] { },
            };
        }
    }

    public class SweepableEstimator<TTrain, TOption> : SweepableEstimatorBase, ISweepableEstimator<TTrain, TOption>
        where TTrain : IEstimator<ITransformer>
        where TOption : class
    {
        internal SweepableEstimator(Func<TOption, TTrain> estimatorFactory, SweepableOption<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything, string estimatorName = null, string[] inputs = null, string[] outputs = null)
            : base(estimatorName, inputs, outputs, optionBuilder.SweepableValueGenerators, scope)
        {
            this.EstimatorFactory = estimatorFactory;
            this.OptionBuilder = optionBuilder;

            if (estimatorName == null)
            {
                this.EstimatorName = estimatorFactory(optionBuilder.CreateDefaultOption()).ToString().Split('.').Last();
            }
            else
            {
                this.EstimatorName = estimatorName;
            }

            this.InputColumns = inputs;
            this.OutputColumns = outputs;
        }

        public SweepableOption<TOption> OptionBuilder { get; private set; }

        public Func<TOption, TTrain> EstimatorFactory { get; private set; }

        public override IEstimator<ITransformer> BuildFromParameters(IDictionary<string, string> parameters)
        {
            var option = this.OptionBuilder.BuildFromParameters(parameters);
            return this.EstimatorFactory(option);
        }

        internal CodeGenNodeContract ToCodeGenNodeContract(Parameters parameters)
        {
            var valueGeneratorIds = this.SweepableValueGenerators.Select(x => x.ID).ToImmutableHashSet();
            var selectedParams = parameters.Where(x => valueGeneratorIds.Contains(x.ID));
            return new CodeGenNodeContract()
            {
                EstimatorName = this.EstimatorName,
                InputColumns = this.InputColumns ?? (new string[] { }),
                OutputColumns = this.OutputColumns ?? new string[] { },
                Parameters = new Parameters(selectedParams),
            };
        }
    }
}
