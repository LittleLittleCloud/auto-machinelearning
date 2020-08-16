// <copyright file="SweepableNodeGenerator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLNet.AutoPipeline
{
    internal class SweepableNodeGenerator : IDiscreteValueGenerator
    {
        private List<INode> nodes;

        public SweepableNodeGenerator(string name, IEnumerable<INode> nodes)
        {
            this.nodes = nodes.ToList();
            this.ID = Guid.NewGuid().ToString("N");
            this.Name = name;
        }

        public SweepableNodeGenerator(string name, INode node)
        {
            this.nodes = new List<INode>()
            {
                node,
            };

            this.ID = Guid.NewGuid().ToString("N");
            this.Name = name;
        }

        public IParameterValue this[int i] => new DiscreteParameterValue(this.Name, this.nodes[i], this.OneHotEncodeValue(this.nodes[i]), this.ID);

        public int Count => this.nodes.Count();

        public string Name { get; set; }

        public string ID { get; private set; }

        public IParameterValue CreateFromNormalized(double normalizedValue)
        {
            return this[(int)(this.Count * normalizedValue)];
        }

        public double[] OneHotEncodeValue(IParameterValue value)
        {
            return this.OneHotEncodeValue(value.RawValue as INode);
        }

        private double[] OneHotEncodeValue(INode node)
        {
            var index = this.nodes.IndexOf(node);
            if (index < 0)
            {
                throw new Exception($"can't find value {node.EstimatorName}");
            }
            else
            {
                var onehot = Enumerable.Repeat(0.0, this.Count).ToArray();
                onehot[index] = 1;
                return onehot;
            }
        }
    }
}
