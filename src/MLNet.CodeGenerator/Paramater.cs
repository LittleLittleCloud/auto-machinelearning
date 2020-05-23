// <copyright file="Paramater.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.CodeGenerator
{
    public class Paramater : ICodeGenNode
    {
        public string ParamaterName { get; private set; }

        public ParamaterValue ParamaterValue { get; private set; }

        public Paramater(string paramaterName, ParamaterValue paramaterValue)
        {
            this.ParamaterName = paramaterName;
            this.ParamaterValue = paramaterValue;
        }

        public string GeneratorCode()
        {
            return $"{this.ParamaterName}={this.ParamaterValue.GeneratorCode()}";
        }
    }
}
