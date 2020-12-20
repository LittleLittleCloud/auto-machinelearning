// <copyright file="OutputEventArgs.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.NNI
{
    public class OutputEventArgs : EventArgs
    {
        public OutputEventArgs(string data)
            : base()
        {
            this.Data = data;
        }

        public string Data { get; }
    }
}
