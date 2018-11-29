#ifndef GRADIENT_HELPERS_H_FSA3FYYR
#define GRADIENT_HELPERS_H_FSA3FYYR

#include <iostream>
#include <map>
#include <string>
#include "Halide.h"

using std::cerr;
using std::endl;

using namespace Halide;

void print_deps(Func F) {
    std::map<std::string, Halide::Internal::Function> flist =
        Halide::Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    cerr << "Dependencies for " << F.name() << " " << endl;
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        cerr << "  .Func " << fit->first << " " << "\n";
    }
}

std::map<std::string, Halide::Internal::Function> get_deps(Func F) {
    std::map<std::string, Halide::Internal::Function> flist =
        Halide::Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    // cerr << "Dependencies for " << F.name() << " " << endl;
    // for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
    //     cerr << "  .Func " << fit->first << " " << "\n";
    // }
    return flist;
}

std::map<std::string, Halide::Internal::Function> get_deps(std::vector<Func> v) {
  std::map<std::string, Halide::Internal::Function> flist;
  for(Func f : v) {
    std::map<std::string, Halide::Internal::Function> curr = get_deps(f);
    flist.insert(curr.begin(), curr.end());
  }
  return flist;
}

void compute_all_root(Func F) {
    std::map<std::string, Halide::Internal::Function> flist =
        Halide::Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        std::vector<Var> args = f.args();
        // cerr << "Func " << f.name() <<" with " << args.size() << " args\n" ;

        f.compute_root();

        // // Vectorize inner most
        // if(args.size() > 0) {
        //   cerr << "arg0 " << args[0].name() << "\n";
        //   Var inner_most = args[0];
        //   // f.vectorize(inner_most, 4);
        // }

    //     // Parallel on all other dims
    //     if(args.size() > 1) {
    //       Var new_var = args[1];
    //         // cerr << "arg " << 1 << " " << args[1].name() << "\n";
    //       for(int i = 2; i < args.size(); ++i) {
    //         // cerr << "arg " << i << " " << args[i].name() << "\n";
    //         f.fuse(new_var, args[i], new_var);
    //       }
    //       f.parallel(new_var);
    //     }
    }
}

void compute_all_at(Func F, Func at_target, Var loc) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(F.function());
    // flist.insert(std::make_pair(F.name(), F.function()));
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        if (f.name() == at_target.name()) {
          continue;
        }
        f.in(at_target).compute_at(at_target, loc);
    }
}

void print_adjoints(std::map<FuncKey, Func> adjoints) {
  for(auto f : adjoints) {
    cerr << f.first.first << " " << f.first.second << "\n";
  }
}

#endif /* end of include guard: GRADIENT_HELPERS_H_FSA3FYYR */
