// <copyright file="SerializableSweeperCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert.Serializable
{
    internal class SerializableSweeperCatalog
    {
        public SerializableSweeperCatalog(MLContext context)
        {
            this.Context = context;
        }

        public MLContext Context { get; }

        public GridSearchSweeper CreateGridSearchSweeper()
        {
            return new GridSearchSweeper();
        }

        public RandomGridSweeper CreateRandomGridSweeper()
        {
            return new RandomGridSweeper();
        }

        public UniformRandomSweeper CreateUniformRandomSweeper(int retry = 10)
        {
            var option = new UniformRandomSweeper.Option()
            {
                Retry = retry,
            };

            return new UniformRandomSweeper(option);
        }
    }
}
