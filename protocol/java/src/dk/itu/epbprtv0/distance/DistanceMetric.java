package dk.itu.epbprtv0.distance;

public interface DistanceMetric<T> {
  class DistancePair {
    public final int index;
    public final double distance;

    public DistancePair(int index, double distance) {
      this.index = index;
      this.distance = distance;
    }

    @Override
    public int hashCode() {
      return index * (1 + (int)distance);
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof DistancePair) {
        DistancePair dp = (DistancePair)o;
        return (index == dp.index && distance == dp.distance);
      } else return false;
    }
  }

  double getDistance(T a, T b);
}