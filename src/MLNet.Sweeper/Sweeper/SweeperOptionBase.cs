// <copyright file="SweeperOptionBase.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime;

namespace MLNet.Sweeper
{
    public class SweeperOptionBase
    {
        public IValueGenerator[] SweptParameters;

        /// <summary>
        /// For RandomSweeper.
        /// </summary>
        public int Retry = 10;
    }
}
