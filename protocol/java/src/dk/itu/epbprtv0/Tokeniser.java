package dk.itu.epbprtv0;

import java.lang.Iterable;
import java.lang.CharSequence;
import java.lang.StringBuilder;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.NoSuchElementException;

public class Tokeniser {
  private static class TokenIterable implements Iterable<String> {
    private class TokenIterator implements Iterator<String> {
      private final int end;
      private int pos = 0;
  
      public TokenIterator(CharSequence input) {
        this.end = (input != null ? input.length() : 0);
      }
      
      private String nt = null;
      private void prime() {
        if (nt != null)
          return;
        if (pos >= end)
          return;
        StringBuilder buf = new StringBuilder();
        boolean force = false;
  
        /* This is a fairly direct port of the logic from frontend.c */
        char c = 0, c1 = 0;
        while (pos < end) {
          c = input.charAt(pos++);
          switch (c) {
            case '\\':
              if (pos != end)
                buf.append(input.charAt(pos++));
              break;
            case '\"':
              force = true;
  dq_string:
              while (pos < end) {
                c1 = input.charAt(pos++);
                switch (c1) {
                  case '\\':
                    if (pos < end) {
                      c1 = input.charAt(pos++);
                      switch (c1) {
                      case '$':
                        /* fall through */
                      case '\\':
                        /* fall through */
                      case '\"':
                        /* fall through */
                        buf.append(c1);
                        break;
                      default:
                        buf.append('\\');
                        buf.append(c1);
                      }
                    }
                    break;
                  case '\"':
                    break dq_string;
                  default:
                    buf.append(c1);
                }
              }
              break;
            case '\'':
              force = true;
  sq_string:
              while (pos < end) {
                c1 = input.charAt(pos++);
                switch (c1) {
                case '\'':
                  break sq_string;
                default:
                  buf.append(c1);
                }
              }
              break;
            case ' ':
              /* fall through */
            case '\t':
              /* fall through */
            case '\n':
              /* fall through */
            case '\r':
              if (force || buf.length() != 0) {
                nt = buf.toString();
                return;
              }
              break;
            default:
              buf.append(c);
          }
        }
        
        if (force || buf.length() != 0) {
          nt = buf.toString();
          return;
        }
      }
  
      @Override
      public boolean hasNext() {
        prime();
        return (nt != null);
      }
  
      @Override
      public String next() {
        prime();
        if (nt != null) {
          String t = nt;
          nt = null;
          return t;
        } else throw new NoSuchElementException();
      }
    }

    private final CharSequence input;
    public TokenIterable(CharSequence input) {
      this.input = input;
    }
    
    @Override
    public Iterator<String> iterator() {
      return new TokenIterator(input);
    }
  }
  public static List<String> tokenise(CharSequence input) {
    ArrayList<String> r = new ArrayList<String>();
    for (String i : new TokenIterable(input))
      r.add(i);
    return r;
  }
}