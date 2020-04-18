// <copyright file="ValueGeneratorOptionBase.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Sweeper
{
    public abstract class ValueGeneratorOptionBase
    {
        public string Name;
    }

    public abstract class NumericValueGeneratorOptionBase : ValueGeneratorOptionBase
    {
        public int NumSteps = 100;

        public double? StepSize = null;

        public bool LogBase = false;
    }
}
