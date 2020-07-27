// <copyright file="UnsweepableNode.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public class UnsweepableNode<TTrainer> : INode
        where TTrainer : IEstimator<ITransformer>
    {
        private TTrainer _instance;
        private TransformerScope _scope;
        private static UnsweepableNode<IEstimator<ITransformer>> no_op = new UnsweepableNode<IEstimator<ITransformer>>();

        public UnsweepableNode(TTrainer instance, TransformerScope scope = TransformerScope.Everything, string estimatorName = null, string[] inputs = null, string[] outputs = null)
        {
            this._instance = instance;
            this._scope = scope;

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

        private UnsweepableNode() { }

        internal static UnsweepableNode<IEstimator<ITransformer>> EmptyNode
        {
            get => no_op;
        }

        public string EstimatorName { get; private set; }

        public TransformerScope Scope => this._scope;

        public IValueGenerator[] ValueGenerators { get => new List<IValueGenerator>().ToArray(); }

        public NodeType NodeType => NodeType.Unsweeapble;

        public string[] InputColumns { get; private set; }

        public string[] OutputColumns { get; private set; }

        public IEstimator<ITransformer> BuildEstimator(ParameterSet parameters)
        {
            return this._instance as IEstimator<ITransformer>;
        }

        public string Summary()
        {
            return $"Node({this.EstimatorName})";
        }

        public override string ToString()
        {
            return this.Summary();
        }

        internal CodeGenNodeContract ToCodeGenNodeContract(ParameterSet parameters = null)
        {
            return new CodeGenNodeContract()
            {
                EstimatorName = this.EstimatorName,
                InputColumns = this.InputColumns ?? (new string[] { }),
                OutputColumns = this.OutputColumns ?? new string[] { },
            };
        }
    }
}
