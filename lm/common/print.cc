#include "lm/common/print.hh"

#include "lm/common/ngram_stream.hh"
#include "util/file_stream.hh"
#include "util/file.hh"
#include "util/mmap.hh"
#include "util/scoped.hh"

#include <sstream>
#include <cstring>

typedef std::map<std::string, std::pair<float, float> > NGramMapWithBackoff;
typedef std::map<std::string, float> NGramMap;

namespace lm {

VocabReconstitute::VocabReconstitute(int fd) {
  uint64_t size = util::SizeOrThrow(fd);
  util::MapRead(util::POPULATE_OR_READ, fd, 0, size, memory_);
  const char *const start = static_cast<const char*>(memory_.get());
  const char *i;
  for (i = start; i != start + size; i += strlen(i) + 1) {
    map_.push_back(i);
  }
  // Last one for LookupPiece.
  map_.push_back(i);
}

namespace {
template <class Payload> std::string PrintLead(const VocabReconstitute &vocab, ProxyStream<Payload> &stream, float* prob) {
  *prob = stream->Value().prob; 
  std::string result = vocab.Lookup(*stream->begin());
  for (const WordIndex *i = stream->begin() + 1; i != stream->end(); ++i) {
    result +=  " " + std::string(vocab.Lookup(*i));
  }
  return result;
}
} // namespace

void PrintARPA::Run(const util::stream::ChainPositions &positions) {
  VocabReconstitute vocab(vocab_fd_);
  util::FileStream out(out_fd_);
  out << "\\data\\\n";
  for (size_t i = 0; i < positions.size(); ++i) {
    out << "ngram " << (i+1) << '=' << counts_[i] << '\n';
  }
  out << '\n';

  for (unsigned order = 1; order < positions.size(); ++order) {
    out << "\\" << order << "-grams:" << '\n';
    NGramMapWithBackoff ngramMap;
    for (ProxyStream<NGram<ProbBackoff> > stream(positions[order - 1], NGram<ProbBackoff>(NULL, order)); stream; ++stream) {
      float prob=0;
      std::string ngram = PrintLead(vocab, stream, &prob);
      if(sort_ngrams_)
        ngramMap[ngram] = std::make_pair(prob, stream->Value().backoff);
      else
        out << prob << '\t' << ngram << '\t' << stream->Value().backoff << '\n';
    }
    if (sort_ngrams_) {
      for (NGramMapWithBackoff::const_iterator n=ngramMap.begin(); n!=ngramMap.end(); ++n)
        out << n->second.first << '\t' << n->first << '\t' << n->second.second << '\n';
      ngramMap.clear();
    }
    out << '\n';
  }

  out << "\\" << positions.size() << "-grams:" << '\n';
  NGramMap ngramMap;
  for (ProxyStream<NGram<Prob> > stream(positions.back(), NGram<Prob>(NULL, positions.size())); stream; ++stream) {
    float prob=0;
    std::string ngram = PrintLead(vocab, stream, &prob);
    if (sort_ngrams_)
      ngramMap[ngram] = prob;
    else
      out << prob << '\t' << ngram << '\n';
  }
  if (sort_ngrams_) {
    for (NGramMap::const_iterator n=ngramMap.begin(); n!=ngramMap.end(); ++n) 
      out << n->second << '\t' << n->first << '\n';
  }
  out << '\n';
  out << "\\end\\\n";
}

} // namespace lm
