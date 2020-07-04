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
    public class UnsweepableNode<TTransformer> : INode
        where TTransformer : ITransformer
    {
        private IEstimator<TTransformer> _instance;
        private TransformerScope _scope;
        private static UnsweepableNode<ITransformer> no_op = new UnsweepableNode<ITransformer>();

        public UnsweepableNode(IEstimator<TTransformer> instance, TransformerScope scope = TransformerScope.Everything, string estimatorName = null)
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
        }

        private UnsweepableNode() { }

        internal static UnsweepableNode<ITransformer> EmptyNode
        {
            get => no_op;
        }

        public ParameterSet Current { get => null; }

        public string EstimatorName { get; private set; }

        public TransformerScope Scope => this._scope;

        public IValueGenerator[] ValueGenerators { get => new List<IValueGenerator>().ToArray(); }

        public NodeType NodeType => NodeType.Unsweeapble;

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
    }
}
