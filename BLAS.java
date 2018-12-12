package com.baileyteam.svm;

/**
 * @author Chris Bailey. Created on 9/30/18.
 */
public final class BLAS {
  static {
    System.loadLibrary("svm");
  }

  public static native void kkt(final int nRows, final int nColumns, final int[] rowIndex, final int[] colBegin,
    final double[] matValue);
}
